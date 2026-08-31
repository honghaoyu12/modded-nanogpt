"""Pure-Python state and math for the opt-in PR328 dual controller.

This module intentionally has no torch or distributed dependencies.  Keeping the
controller state here makes its invariants testable without constructing the
Track 3 model or starting a training process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


@dataclass(frozen=True)
class ControllerConfig:
    initial_multiplier: float = 1.0
    global_multiplier_min: float = 0.85
    global_multiplier_max: float = 1.20
    muon_multiplier_min: float = 0.75
    muon_multiplier_max: float = 1.35
    nonmuon_multiplier_min: float = 0.75
    nonmuon_multiplier_max: float = 1.35
    global_kp: float = 0.04
    allocation_kp: float = 0.005
    rho_target: float = 0.91
    rho_target_early: float | None = None
    rho_target_cruise: float | None = None
    rho_target_tail: float | None = None
    rho_early_end: int = 300
    rho_cruise_end: int = 2200
    rho_ramp_steps: int = 300
    rho_beta: float = 0.9
    rho_clip_min: float = -1.0
    rho_clip_max: float = 3.0
    factor_min: float = 0.9
    factor_max: float = 1.1
    rho_deadband: float = 0.0
    allocation_log_bound: float = 0.10
    calibration_probes: int = 20
    component_min_contribution: float = 1e-12
    interaction_max: float = 0.25
    sigma_floor: float = 0.05


class DualMultiplierController:
    """Hierarchical global-authority plus Muon/non-Muon allocation controller.

    ``global_log_scale`` controls the shared authority.  ``allocation_log_scale``
    reallocates that authority while preserving the contribution-weighted log
    mean whenever family clipping is inactive.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._validate_config()
        initial = _clip(
            config.initial_multiplier,
            config.global_multiplier_min,
            config.global_multiplier_max,
        )
        self.global_log_scale = math.log(initial)
        self.allocation_log_scale = 0.0
        self.rho_ema_total: float | None = None
        self.update_count = 0
        self.weights = (0.5, 0.5)
        self.calibration_samples: list[tuple[float, float]] = []
        self.calibration_center = (0.0, 0.0)
        self.calibration_scale = (config.sigma_floor, config.sigma_floor)
        self.calibration_complete = config.calibration_probes == 0
        self.allocation_frozen = False
        self.allocation_freeze_reason = "" if not self.calibration_complete else ""

    def _validate_config(self) -> None:
        c = self.config
        if not 0 < c.global_multiplier_min <= c.global_multiplier_max:
            raise ValueError("invalid global multiplier bounds")
        if c.calibration_probes > 0 and not math.isclose(c.initial_multiplier, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("passive calibration requires initial multiplier 1.0")
        if not 0 < c.muon_multiplier_min <= c.muon_multiplier_max:
            raise ValueError("invalid Muon multiplier bounds")
        if not 0 < c.nonmuon_multiplier_min <= c.nonmuon_multiplier_max:
            raise ValueError("invalid non-Muon multiplier bounds")
        if not 0 < c.factor_min <= c.factor_max:
            raise ValueError("invalid controller factor bounds")
        if not c.rho_clip_min < c.rho_clip_max:
            raise ValueError("invalid rho clipping bounds")
        if not 0 <= c.rho_ramp_steps:
            raise ValueError("rho ramp steps must be non-negative")
        if c.rho_early_end < 0 or c.rho_cruise_end < c.rho_early_end:
            raise ValueError("invalid rho phase boundaries")
        phase_targets = (c.rho_target_early, c.rho_target_cruise, c.rho_target_tail)
        if any(value is not None for value in phase_targets) and not all(value is not None for value in phase_targets):
            raise ValueError("phase rho targets must be provided together")
        if c.calibration_probes < 0 or c.component_min_contribution < 0:
            raise ValueError("invalid component calibration settings")
        if c.interaction_max < 0 or c.sigma_floor <= 0:
            raise ValueError("invalid component signal settings")

    def rho_target_for_step(self, step: int) -> float:
        c = self.config
        if c.rho_target_early is None:
            return c.rho_target
        early = c.rho_target_early
        cruise = c.rho_target_cruise
        tail = c.rho_target_tail
        assert cruise is not None and tail is not None
        ramp = c.rho_ramp_steps
        if step <= c.rho_early_end:
            return early
        if ramp > 0 and step < c.rho_early_end + ramp:
            progress = (step - c.rho_early_end) / ramp
            return early + progress * (cruise - early)
        if step <= c.rho_cruise_end:
            return cruise
        if ramp > 0 and step < c.rho_cruise_end + ramp:
            progress = (step - c.rho_cruise_end) / ramp
            return cruise + progress * (tail - cruise)
        return tail

    def multipliers(self) -> dict[str, float]:
        """Return current family multipliers using the latest contribution shares."""
        w_muon, w_nonmuon = self.weights
        d = _clip(
            self.allocation_log_scale,
            -self.config.allocation_log_bound,
            self.config.allocation_log_bound,
        )
        muon_log = self.global_log_scale + w_nonmuon * d
        nonmuon_log = self.global_log_scale - w_muon * d
        muon_unclipped = math.exp(muon_log)
        nonmuon_unclipped = math.exp(nonmuon_log)
        muon = _clip(muon_unclipped, self.config.muon_multiplier_min, self.config.muon_multiplier_max)
        nonmuon = _clip(nonmuon_unclipped, self.config.nonmuon_multiplier_min, self.config.nonmuon_multiplier_max)
        return {
            "muon": muon,
            "nonmuon": nonmuon,
            "global_log_scale": self.global_log_scale,
            "allocation_log_scale": self.allocation_log_scale,
            "bound_hit_muon": int(muon != muon_unclipped),
            "bound_hit_nonmuon": int(nonmuon != nonmuon_unclipped),
        }

    def _valid_total_signal(self, *, actual: float, predicted: float) -> bool:
        denominator_floor = max(1e-12, abs(float(actual)) * 1e-12)
        return _finite(actual) and _finite(predicted) and predicted > denominator_floor

    def _update_global(self, *, step: int, actual: float, predicted: float) -> dict[str, object]:
        c = self.config
        valid = self._valid_total_signal(actual=actual, predicted=predicted)
        invalid_reason = ""
        rho: float | None = None
        factor = 1.0
        if not valid:
            invalid_reason = "nonpositive_or_nonfinite_predicted_decrease"
        else:
            rho = (float(actual) / predicted)
            if not _finite(rho):
                valid = False
                invalid_reason = "nonfinite_rho"
            else:
                rho_clipped = _clip(rho, c.rho_clip_min, c.rho_clip_max)
                if self.rho_ema_total is None:
                    self.rho_ema_total = rho_clipped
                else:
                    self.rho_ema_total = c.rho_beta * self.rho_ema_total + (1.0 - c.rho_beta) * rho_clipped
                error = self.rho_ema_total - self.rho_target_for_step(step)
                raw_factor = 1.0 if abs(error) <= c.rho_deadband else math.exp(c.global_kp * error)
                factor = _clip(raw_factor, c.factor_min, c.factor_max)
                next_global = self.global_log_scale + math.log(factor)
                self.global_log_scale = _clip(
                    next_global,
                    math.log(c.global_multiplier_min),
                    math.log(c.global_multiplier_max),
                )
                self.update_count += 1
        return {
            "rho_total": "" if rho is None else rho,
            "rho_ema_total": "" if self.rho_ema_total is None else self.rho_ema_total,
            "rho_target": self.rho_target_for_step(step),
            "global_update_factor": factor,
            "global_valid": int(valid),
            "global_invalid_reason": invalid_reason,
        }

    def _finish_calibration(self) -> None:
        if self.calibration_complete or len(self.calibration_samples) < self.config.calibration_probes:
            return
        muon = sorted(sample[0] for sample in self.calibration_samples)
        nonmuon = sorted(sample[1] for sample in self.calibration_samples)
        midpoint = len(muon) // 2
        centers = (
            muon[midpoint] if len(muon) % 2 else (muon[midpoint - 1] + muon[midpoint]) / 2,
            nonmuon[midpoint] if len(nonmuon) % 2 else (nonmuon[midpoint - 1] + nonmuon[midpoint]) / 2,
        )
        deviations = [
            sorted(abs(value - centers[0]) for value in muon),
            sorted(abs(value - centers[1]) for value in nonmuon),
        ]
        scales = []
        for values in deviations:
            mid = len(values) // 2
            mad = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
            scales.append(max(self.config.sigma_floor, 1.4826 * mad))
        self.calibration_center = centers
        self.calibration_scale = (scales[0], scales[1])
        self.calibration_complete = True
        self.calibration_samples.clear()

    def observe(
        self,
        *,
        step: int,
        actual_full: float,
        predicted_muon: float,
        predicted_nonmuon: float,
        actual_muon: float | None = None,
        actual_nonmuon: float | None = None,
        interaction_residual: float | None = None,
    ) -> dict[str, object]:
        """Consume one full observation and optionally a component probe."""
        predicted_muon = float(predicted_muon)
        predicted_nonmuon = float(predicted_nonmuon)
        predicted_total = predicted_muon + predicted_nonmuon
        actual_full = float(actual_full)
        total_valid = self._valid_total_signal(actual=actual_full, predicted=predicted_total)
        shares_valid = (
            _finite(predicted_muon)
            and _finite(predicted_nonmuon)
            and predicted_total > self.config.component_min_contribution
        )
        if shares_valid and total_valid and self.calibration_complete:
            positive_muon = max(predicted_muon, 0.0)
            positive_nonmuon = max(predicted_nonmuon, 0.0)
            positive_total = positive_muon + positive_nonmuon
            if positive_total > self.config.component_min_contribution:
                self.weights = (positive_muon / positive_total, positive_nonmuon / positive_total)
            else:
                shares_valid = False

        multipliers_before = self.multipliers()
        if self.calibration_complete:
            result = self._update_global(
                step=step,
                actual=actual_full,
                predicted=predicted_total,
            )
        else:
            result = {
                "rho_total": "",
                "rho_ema_total": "",
                "rho_target": self.rho_target_for_step(step),
                "global_update_factor": 1.0,
                "global_valid": 0,
                "global_invalid_reason": "calibrating",
            }
        component_valid = False
        allocation_update = 0.0
        allocation_freeze_reason = ""
        rho_muon: float | None = None
        rho_nonmuon: float | None = None
        if actual_muon is not None and actual_nonmuon is not None:
            muon_valid = _finite(actual_muon) and _finite(predicted_muon) and predicted_muon > self.config.component_min_contribution
            nonmuon_valid = _finite(actual_nonmuon) and _finite(predicted_nonmuon) and predicted_nonmuon > self.config.component_min_contribution
            if muon_valid:
                rho_muon = float(actual_muon) / predicted_muon
            if nonmuon_valid:
                rho_nonmuon = float(actual_nonmuon) / predicted_nonmuon
            component_valid = total_valid and muon_valid and nonmuon_valid and _finite(rho_muon) and _finite(rho_nonmuon)
            if component_valid and interaction_residual is not None:
                interaction_scale = max(abs(float(actual_full)), self.config.component_min_contribution)
                if abs(float(interaction_residual)) / interaction_scale > self.config.interaction_max:
                    component_valid = False
                    allocation_freeze_reason = "interaction_residual_too_large"
            if component_valid and not self.calibration_complete:
                self.calibration_samples.append((rho_muon, rho_nonmuon))
                self._finish_calibration()
                allocation_freeze_reason = "calibrating" if not self.calibration_complete else ""
            elif component_valid and self.calibration_complete:
                muon_z = (rho_muon - self.calibration_center[0]) / max(self.calibration_scale[0], self.config.sigma_floor)
                nonmuon_z = (rho_nonmuon - self.calibration_center[1]) / max(self.calibration_scale[1], self.config.sigma_floor)
                error = muon_z - nonmuon_z
                if abs(error) <= self.config.rho_deadband:
                    error = 0.0
                allocation_update = self.config.allocation_kp * error
                self.allocation_log_scale = _clip(
                    self.allocation_log_scale + allocation_update,
                    -self.config.allocation_log_bound,
                    self.config.allocation_log_bound,
                )
            else:
                allocation_freeze_reason = allocation_freeze_reason or "invalid_component_signal"
        elif not shares_valid:
            allocation_freeze_reason = "invalid_predicted_contributions"

        if not self.calibration_complete and not allocation_freeze_reason:
            allocation_freeze_reason = "calibrating"
        self.allocation_frozen = bool(allocation_freeze_reason)
        self.allocation_freeze_reason = allocation_freeze_reason
        multipliers = self.multipliers()
        result.update(
            {
                "predicted_decrease_total": predicted_total,
                "predicted_decrease_muon": predicted_muon,
                "predicted_decrease_nonmuon": predicted_nonmuon,
                "muon_contribution_fraction": self.weights[0],
                "nonmuon_contribution_fraction": self.weights[1],
                "allocation_update": allocation_update,
                "allocation_frozen": int(self.allocation_frozen),
                "allocation_freeze_reason": self.allocation_freeze_reason,
                "rho_muon": "" if rho_muon is None else rho_muon,
                "rho_nonmuon": "" if rho_nonmuon is None else rho_nonmuon,
                "component_signal_validity": int(component_valid),
                "calibration_complete": int(self.calibration_complete),
                "calibration_center_muon": self.calibration_center[0],
                "calibration_center_nonmuon": self.calibration_center[1],
                "calibration_scale_muon": self.calibration_scale[0],
                "calibration_scale_nonmuon": self.calibration_scale[1],
                "interaction_residual": "" if interaction_residual is None else float(interaction_residual),
                "multiplier_muon": multipliers_before["muon"],
                "multiplier_nonmuon": multipliers_before["nonmuon"],
                "multiplier_next_muon": multipliers["muon"],
                "multiplier_next_nonmuon": multipliers["nonmuon"],
                **multipliers,
            }
        )
        return result
