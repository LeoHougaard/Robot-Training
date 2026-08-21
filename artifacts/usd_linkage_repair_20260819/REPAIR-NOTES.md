# RF linkage repair

## Inputs

- `Isaac lab Assembly.zip`, exported at 2026-08-19 09:50. It contains seven
  revolute joints and leaves both closed linkages open.
- `Isaac lab Assembly(1).zip`, exported at 2026-08-19 09:54. It contains all
  nine revolute joints, but adds both closed loops directly to a
  reduced-coordinate articulation.

The geometry and full joint definitions come from the second export. Neither
input ZIP was modified.

## Repair

The repaired USD keeps seven revolute joints in a connected, acyclic
articulation tree. These two passive joints remain active Physics constraints
but have `physics:excludeFromArticulation = 1`:

- `Femure Tibula RF`, prim `_M5Wr42GBylD274sgD`
- `Hub Link RF`, prim `_M0vdxvTALZGcsulyT`

Those are the two source mates whose initial anchors disagree, by 0.700 mm and
4.589 mm respectively. Breaking the articulation loops there leaves all seven
tree-joint anchors coincident at the exported pose. PhysX can solve the two
remaining gaps as maximal-coordinate loop closures.

The chassis-to-world fixed joint was removed. `PhysicsArticulationRootAPI` and
`PhysxArticulationAPI` are applied to `/robot`, which leaves the robot floating
for simulation and training.

Only the three joints marked Driven in `robot.json` have a Drive API:

| Joint | Stiffness | Damping | Maximum torque | Maximum velocity | Armature |
|---|---:|---:|---:|---:|---:|
| `Main Hub RF` | 22 | 0.8 | 1.37 N m | 8 rad/s | 0.001 kg m2 |
| `Hub Tibula RF` | 22 | 0.8 | 1.37 N m | 8 rad/s | 0.001 kg m2 |
| `Hub Servo Link RF` | 22 | 0.8 | 1.37 N m | 8 rad/s | 0.001 kg m2 |

The velocity authored in USD is `458.36624` degrees per second, which is
8 radians per second. Passive joints no longer have `PhysicsDriveAPI:angular`.

## Expected graph

- 8 rigid bodies
- 9 revolute constraints total
- 7 revolute joints in the articulation tree
- 2 excluded loop-closure constraints
- 3 driven joints
- 0 fixed-to-world joints

The repaired directory contains the editable files. The ZIP at the artifact
root contains `robot.usda`, `robot.json`, and `meshes/` in the same layout as
the Publisher exports.
