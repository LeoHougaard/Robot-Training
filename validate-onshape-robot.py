"""Open an Onshape Publisher USD in Isaac Sim and validate its robot graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True, help="Absolute container path to robot.usda")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdPhysics

    asset = Path(args.asset)
    if not asset.is_absolute() or asset.name != "robot.usda" or not asset.is_file():
        raise RuntimeError(f"Expected an existing absolute robot.usda path, got: {asset}")

    context = omni.usd.get_context()
    print(f"opening_asset={asset}", flush=True)
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError("Isaac Sim could not open the Publisher USD.")
    print(f"opened_default_prim={stage.GetDefaultPrim().GetPath()}", flush=True)

    revolute_joints = []
    articulation_revolute_joints = []
    loop_closure_joints = []
    rigid_bodies = []
    collisions = []
    robot_roots = []
    broken_joint_targets = []

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joints.append(prim)
            joint = UsdPhysics.Joint(prim)
            if joint.GetExcludeFromArticulationAttr().Get():
                loop_closure_joints.append(prim)
            else:
                articulation_revolute_joints.append(prim)
            for relation_name, relation in (
                ("body0", joint.GetBody0Rel()),
                ("body1", joint.GetBody1Rel()),
            ):
                targets = relation.GetTargets()
                if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]).IsValid():
                    broken_joint_targets.append(f"{prim.GetPath()}:{relation_name}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(prim)
        applied_schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        authored_api_schemas = str(prim.GetMetadata("apiSchemas") or "")
        if "IsaacRobotAPI" in applied_schemas or "IsaacRobotAPI" in authored_api_schemas:
            robot_roots.append(prim)

    if not robot_roots:
        raise RuntimeError("No IsaacRobotAPI root was found.")
    if not revolute_joints:
        raise RuntimeError("No revolute joints were found.")
    if broken_joint_targets:
        raise RuntimeError(
            "Broken joint body relationships: " + ", ".join(broken_joint_targets)
        )
    if len(rigid_bodies) < len(articulation_revolute_joints) + 1:
        raise RuntimeError(
            f"Found {len(rigid_bodies)} rigid bodies for "
            f"{len(articulation_revolute_joints)} articulation revolute joints."
        )

    # A reduced-coordinate articulation must be a tree. Closed linkages remain
    # valid when each extra loop-closing joint is explicitly excluded from the
    # articulation and solved as a maximal-coordinate constraint by PhysX.
    parents = {}

    def find(body):
        parents.setdefault(body, body)
        while parents[body] != body:
            parents[body] = parents[parents[body]]
            body = parents[body]
        return body

    def union(body0, body1):
        root0, root1 = find(body0), find(body1)
        if root0 == root1:
            return False
        parents[root1] = root0
        return True

    articulation_cycles = []
    for prim in articulation_revolute_joints:
        joint = UsdPhysics.Joint(prim)
        body0 = str(joint.GetBody0Rel().GetTargets()[0])
        body1 = str(joint.GetBody1Rel().GetTargets()[0])
        if not union(body0, body1):
            articulation_cycles.append(str(prim.GetPath()))
    if articulation_cycles:
        raise RuntimeError(
            "Reduced-coordinate articulation contains cycles: "
            + ", ".join(articulation_cycles)
        )

    invalid_closures = []
    for prim in loop_closure_joints:
        joint = UsdPhysics.Joint(prim)
        body0 = str(joint.GetBody0Rel().GetTargets()[0])
        body1 = str(joint.GetBody1Rel().GetTargets()[0])
        if find(body0) != find(body1):
            invalid_closures.append(str(prim.GetPath()))
    if invalid_closures:
        raise RuntimeError(
            "Excluded joints do not close an articulation loop: "
            + ", ".join(invalid_closures)
        )
    if not collisions:
        raise RuntimeError("No collision-enabled prims were found.")
    print("robot_graph_assertions=PASS", flush=True)

    # Reference the validated asset into the application's disposable runtime
    # stage and play a few headless frames so PhysX consumes the composition.
    runtime_stage = context.get_stage()
    if runtime_stage is None:
        context.new_stage()
        simulation_app.update()
        runtime_stage = context.get_stage()
    runtime_stage.DefinePrim("/World", "Xform")
    robot_prim = runtime_stage.DefinePrim("/World/Robot", "Xform")
    robot_prim.GetReferences().AddReference(str(asset))
    if not any(prim.IsA(UsdPhysics.Scene) for prim in runtime_stage.Traverse()):
        UsdPhysics.Scene.Define(runtime_stage, "/PhysicsScene")
    for _ in range(3):
        simulation_app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(12):
        simulation_app.update()
    timeline.stop()
    print("physx_frames=12", flush=True)

    used_layers = [layer.identifier for layer in stage.GetUsedLayers()]
    print("ONSHAPE_ROBOT_VALIDATION=PASS", flush=True)
    print(f"asset={asset}", flush=True)
    print(f"default_prim={stage.GetDefaultPrim().GetPath()}", flush=True)
    print(f"isaac_robot_roots={len(robot_roots)}", flush=True)
    print(f"revolute_joints={len(revolute_joints)}", flush=True)
    print(
        f"articulation_revolute_joints={len(articulation_revolute_joints)}",
        flush=True,
    )
    print(f"loop_closure_joints={len(loop_closure_joints)}", flush=True)
    print(f"rigid_bodies={len(rigid_bodies)}", flush=True)
    print(f"collision_prims={len(collisions)}", flush=True)
    print(f"composed_layers={len(used_layers)}", flush=True)
except Exception:
    print("ONSHAPE_ROBOT_VALIDATION=FAIL", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid SimulationApp.close() masking the validator's non-zero status.
    os._exit(1)
finally:
    simulation_app.close()
