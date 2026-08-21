"""Unit tests for portable-policy deployment metadata."""

from __future__ import annotations

import math
import unittest

from export_v2_policy import (
    REQUIRED_GOAL_COMMANDS,
    deployment_contract,
    validate_goal_evaluation,
    validate_robust_test_evaluation,
)


class DeploymentContractTests(unittest.TestCase):
    def contract(self, **overrides):
        values = {
            "forward_min": -0.18,
            "forward_max": 0.22,
            "lateral_min": -0.16,
            "lateral_max": 0.16,
            "yaw_max": 0.25,
            "planar_deadband": 0.02,
            "yaw_deadband": 0.03,
            "stance_action": [0.0] * 12,
            "action_limit_by_joint": [0.5, 1.0, 1.0] * 4,
            "action_filter_alpha": 0.2,
            "action_delta_limit": 0.2,
            "position_target_scale_rad": 0.3,
            "command_smoothing_time_s": 0.4,
        }
        values.update(overrides)
        return deployment_contract(**values)

    def test_full_mobility_limits_are_exported(self):
        limits, stationary, action = self.contract()
        self.assertEqual(limits["forward_m_s"], [-0.18, 0.22])
        self.assertEqual(limits["lateral_m_s"], [-0.16, 0.16])
        self.assertEqual(limits["yaw_rate_rad_s"], [-0.25, 0.25])
        self.assertEqual(len(stationary["normalized_stance_action"]), 12)
        self.assertEqual(
            action["applied_normalized_clip_by_joint"], [0.5, 1.0, 1.0] * 4
        )
        self.assertEqual(action["low_pass_alpha"], 0.2)
        self.assertEqual(action["applied_normalized_slew_limit"], 0.2)
        self.assertEqual(action["position_target_scale_rad"], 0.3)

    def test_rejects_limits_beyond_the_promotion_screen(self):
        with self.assertRaisesRegex(ValueError, "lateral maximum"):
            self.contract(lateral_max=0.17)

    def test_rejects_non_finite_or_out_of_range_stance(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.contract(stance_action=[0.0] * 11 + [math.nan])
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.contract(stance_action=[0.0] * 11 + [1.1])

    def test_rejects_invalid_per_joint_action_limits(self):
        with self.assertRaisesRegex(ValueError, "Per-joint"):
            self.contract(action_limit_by_joint=[1.0] * 11 + [0.0])

    def test_rejects_invalid_action_filter(self):
        with self.assertRaisesRegex(ValueError, "filter alpha"):
            self.contract(action_filter_alpha=0.0)

    def test_rejects_invalid_action_slew_and_position_scale(self):
        with self.assertRaisesRegex(ValueError, "slew limit"):
            self.contract(action_delta_limit=0.0)
        with self.assertRaisesRegex(ValueError, "Position target scale"):
            self.contract(position_target_scale_rad=0.0)
        with self.assertRaisesRegex(ValueError, "Command smoothing time"):
            self.contract(command_smoothing_time_s=0.0)

    def test_rejects_stale_goal_evaluation(self):
        evaluation = {
            "stage": "goal",
            "passed": True,
            "segments": {"stand": {}},
        }
        with self.assertRaisesRegex(ValueError, "predates the full mobility"):
            validate_goal_evaluation(evaluation)

    def test_accepts_the_fixed_goal_screen(self):
        segments = {
            name: {
                "command_forward": command[0],
                "command_lateral": command[1],
                "command_yaw": command[2],
            }
            for name, command in REQUIRED_GOAL_COMMANDS.items()
        }
        validate_goal_evaluation(
            {"stage": "goal", "passed": True, "segments": segments}
        )

    def test_robust_test_export_still_requires_a_passing_robust_screen(self):
        validate_robust_test_evaluation(
            {"stage": "robust", "passed": True, "segments": {"forward": {}}}
        )
        with self.assertRaisesRegex(ValueError, "passing evaluation"):
            validate_robust_test_evaluation(
                {"stage": "robust", "passed": False, "segments": {"forward": {}}}
            )
        with self.assertRaisesRegex(ValueError, "passing Robust"):
            validate_robust_test_evaluation(
                {"stage": "goal", "passed": True, "segments": {"forward": {}}}
            )


if __name__ == "__main__":
    unittest.main()
