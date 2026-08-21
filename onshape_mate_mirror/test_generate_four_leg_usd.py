import json
import tempfile
import unittest
from pathlib import Path

import generate_four_leg_usd as generator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ASSET = REPOSITORY_ROOT / "artifacts" / "usd_linkage_repair_20260819" / "fixed"
VERIFIER_TEMPLATE = Path(__file__).with_name("four_leg_verifier_template.html")


class FourLegGeneratorTests(unittest.TestCase):
    def test_generates_the_verified_robot_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checks = generator.generate(SOURCE_ASSET, output, VERIFIER_TEMPLATE)

            self.assertEqual(checks["rigidBodies"], 29)
            self.assertEqual(checks["revoluteJoints"], 36)
            self.assertEqual(checks["articulationTreeJoints"], 28)
            self.assertEqual(checks["loopClosures"], 8)
            self.assertEqual(checks["drives"], 12)
            self.assertAlmostEqual(checks["lengthMm"], 188.22341, places=6)
            self.assertAlmostEqual(checks["widthMm"], 180.0, places=6)
            self.assertTrue(checks["allRigidTransformsProper"])
            self.assertTrue(checks["allJointAxesAligned"])
            self.assertTrue(checks["allJointFramesAligned"])
            self.assertLess(checks["maxTreeAnchorMismatchMm"], 0.0001)
            self.assertLess(max(checks["loopAnchorMismatchMm"]), 0.0001)

            for filename in (
                "robot.usda",
                "robot.json",
                "four-leg-visual.gltf",
                "verify-four-leg.html",
                "VERIFICATION.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)

            manifest = json.loads((output / "robot.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["Corners"], ["RF", "RR", "LF", "LR"])
            self.assertEqual(len(manifest["RigidBodies"]), 29)
            self.assertEqual(len(manifest["Joints"]), 36)


if __name__ == "__main__":
    unittest.main()
