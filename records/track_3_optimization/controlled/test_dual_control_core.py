import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dual_control_core import ControllerConfig, DualMultiplierController


class DualControllerCoreTest(unittest.TestCase):
    def test_weighted_log_authority_is_preserved_without_clipping(self):
        controller = DualMultiplierController(ControllerConfig(
            global_multiplier_min=0.1,
            global_multiplier_max=10.0,
            muon_multiplier_min=0.1,
            muon_multiplier_max=10.0,
            nonmuon_multiplier_min=0.1,
            nonmuon_multiplier_max=10.0,
            allocation_log_bound=1.0,
        ))
        controller.global_log_scale = math.log(1.1)
        controller.allocation_log_scale = 0.2
        controller.weights = (0.7, 0.3)
        multipliers = controller.multipliers()
        weighted_log = 0.7 * math.log(multipliers["muon"]) + 0.3 * math.log(multipliers["nonmuon"])
        self.assertAlmostEqual(weighted_log, math.log(1.1), places=12)

    def test_equal_initial_authority_is_native(self):
        controller = DualMultiplierController(ControllerConfig())
        multipliers = controller.multipliers()
        self.assertEqual(multipliers["muon"], 1.0)
        self.assertEqual(multipliers["nonmuon"], 1.0)

    def test_invalid_total_signal_freezes_state(self):
        controller = DualMultiplierController(ControllerConfig(calibration_probes=0))
        controller.allocation_log_scale = 0.1
        controller.weights = (0.2, 0.8)
        before = (controller.global_log_scale, controller.allocation_log_scale, controller.weights, controller.multipliers())
        result = controller.observe(
            step=10,
            actual_full=float("nan"),
            predicted_muon=1.0,
            predicted_nonmuon=1.0,
        )
        after = (controller.global_log_scale, controller.allocation_log_scale, controller.weights, controller.multipliers())
        self.assertEqual(after, before)
        self.assertEqual(result["global_valid"], 0)

        result = controller.observe(
            step=11, actual_full=float("nan"), predicted_muon=1.0, predicted_nonmuon=1.0,
            actual_muon=0.5, actual_nonmuon=0.5, interaction_residual=0.0,
        )
        self.assertEqual(controller.allocation_log_scale, before[1])
        self.assertEqual(result["component_signal_validity"], 0)

    def test_passive_calibration_keeps_authority_at_one(self):
        controller = DualMultiplierController(ControllerConfig(calibration_probes=2))
        for step in (0, 1):
            controller.observe(
                step=step, actual_full=0.2, predicted_muon=0.1, predicted_nonmuon=0.1,
                actual_muon=0.1, actual_nonmuon=0.1, interaction_residual=0.0,
            )
            self.assertEqual(controller.multipliers()["muon"], 1.0)
            self.assertEqual(controller.multipliers()["nonmuon"], 1.0)
            self.assertEqual(controller.update_count, 0)

    def test_nonunit_initial_multiplier_rejected_for_passive_calibration(self):
        with self.assertRaisesRegex(ValueError, "initial multiplier 1.0"):
            DualMultiplierController(ControllerConfig(initial_multiplier=1.1, calibration_probes=1))

    def test_component_calibration_then_allocation_update(self):
        controller = DualMultiplierController(ControllerConfig(
            calibration_probes=2,
            allocation_kp=0.1,
        ))
        for step in (0, 1):
            controller.observe(
                step=step,
                actual_full=0.2,
                predicted_muon=0.1,
                predicted_nonmuon=0.1,
                actual_muon=0.1,
                actual_nonmuon=0.1,
                interaction_residual=0.0,
            )
        self.assertTrue(controller.calibration_complete)
        old_d = controller.allocation_log_scale
        result = controller.observe(
            step=2,
            actual_full=0.2,
            predicted_muon=0.1,
            predicted_nonmuon=0.1,
            actual_muon=0.2,
            actual_nonmuon=0.1,
            interaction_residual=0.0,
        )
        self.assertGreater(controller.allocation_log_scale, old_d)
        self.assertEqual(result["component_signal_validity"], 1)

    def test_large_interaction_freezes_allocation(self):
        controller = DualMultiplierController(ControllerConfig(
            calibration_probes=0,
            allocation_kp=0.1,
            interaction_max=0.25,
        ))
        before = controller.allocation_log_scale
        result = controller.observe(
            step=0,
            actual_full=0.2,
            predicted_muon=0.1,
            predicted_nonmuon=0.1,
            actual_muon=0.1,
            actual_nonmuon=0.1,
            interaction_residual=0.1,
        )
        self.assertEqual(controller.allocation_log_scale, before)
        self.assertEqual(result["allocation_freeze_reason"], "interaction_residual_too_large")


if __name__ == "__main__":
    unittest.main()
