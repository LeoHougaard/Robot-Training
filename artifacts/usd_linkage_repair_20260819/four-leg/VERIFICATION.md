# Four-leg generation verification

- Rigid bodies: 29 (1 chassis + 4 x 7 leg bodies)
- Revolute constraints: 36 (4 x 9)
- Articulation-tree joints: 28 (4 x 7)
- Independent loop closures: 8 (4 x 2)
- Driven joints: 12 (4 x 3)
- Main Hub length spacing: 188.22341 mm
- Main Hub width spacing: 180.00000 mm
- Every rigid-body transform has determinant +1 (no reflected physics frames).
- Maximum articulation-tree anchor mismatch: 0.000001166 mm.
- Both local frames of every revolute joint have aligned axes.
- Both loop-closure pivots per leg are coincident at the authored pose (no initial constraint preload).
- RR and LF meshes are locally mirrored with triangle winding and normals corrected.
- RR and LF revolute limits are sign-inverted and swapped; LR limits retain RF signs.
- Visual verification: open `verify-four-leg.html` or import `four-leg-visual.gltf` into Onshape.
- The glTF verifies geometry only; USD physics joints remain in `robot.usda`.
