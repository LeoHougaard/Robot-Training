# Zero-API Onshape leg duplication

Use Onshape's native Assembly duplication for robot legs. Do not use Assembly
Mirror for the finished mechanism: Assembly Mirror deliberately makes mirrored
motion follow the source and does not create independent mates.

This workflow makes independent revolute, cylindrical, slider, fastened, and
other mates without FeatureScript, REST calls, or API keys. The included mate
name helper drives Onshape's visible **Rename** command in your browser; it does
not read or write the Onshape API.

## Why this works

Onshape supports all of the required operations natively:

- **Duplicate** on an Assembly tab copies that assembly in the current document
  while maintaining its references. The complete mate feature list is part of
  the duplicated assembly.
- **Replace instance** reapplies existing mates when a component is replaced,
  including when the replacement is a mirrored derivative.
- Mates inside an inserted subassembly remain active and flexible in the parent
  assembly.
- **Dissolve subassembly** moves all of its instances and mate connections into
  the parent assembly. This produces the flat, independent mate list needed by
  the current Onshape Labs Omniverse Publisher workflow.

Because every corner gets its own duplicated Assembly tab, editing a mate or
moving one leg cannot drive another leg.

## One-time preparation

Use the working leg in the screenshot as the template. It currently contains
eight instances and nine mates.

1. Rename its Assembly tab to `Leg-RF-template`.
2. Confirm all nine mates solve and animate correctly. Cylindrical mates are
   valid; they do not need to be converted to revolute mates.
3. Give the mounting part (`Main Hub` in the screenshot) an explicit, reusable
   Mate connector at the chassis attachment point. Prefer a Part Studio Mate
   connector because it is available on every instance of the part.
4. Do not depend on `Fix` for attachment to the dog body. Fixing an instance is
   local to its current assembly and does not carry into the parent assembly.
5. Create a version named something like `Leg template validated` before making
   the export copies.

## Create the four independent legs

Right-click the `Leg-RF-template` Assembly tab and choose **Duplicate** three
times. Rename the four tabs:

- `Leg-RF-export`
- `Leg-RR-export`
- `Leg-LF-export`
- `Leg-LR-export`

These are four separate mate graphs. The nine mates, including cylindrical
mates, are already present in every copy.

## Rename the copied mates

Duplicating the RF Assembly preserves its mate names, so each mate initially
still ends in `RF`. Before inserting or dissolving the copies, use
[`rename-mate-suffix.user.js`](./rename-mate-suffix.user.js) in each Assembly:

| Assembly tab | Mate-name action |
|---|---|
| `Leg-RF-export` | Leave `RF` unchanged |
| `Leg-RR-export` | Rename every final `RF` to `RR` |
| `Leg-LF-export` | Rename every final `RF` to `LF` |
| `Leg-LR-export` | Rename every final `RF` to `LR` |

For example, `Hub Tibula RF` becomes `Hub Tibula LF` in the LF copy. The
renamer only scans rows below the active Assembly's **Mate features** heading;
instance names are not included.

### Run the helper yourself

This is intentionally a user-run browser helper. No API credentials or Codex
session are required.

1. Save the script in a browser user-script manager, or copy its body into a
   Chromium DevTools **Sources > Snippets** snippet. If using a snippet, omit
   only the initial `// ==UserScript==` metadata block.
2. Open one duplicated leg Assembly, clear the tree filter, and expand
   **Mate features** so all mate rows are loaded.
3. Run the snippet. A **Rename RF mates** button appears at the lower right.
4. Click it and choose `RR`, `LF`, or `LR`. The target defaults from the active
   tab name when that tab follows the `Leg-*-export` naming above.
5. Review the complete old-to-new preview, then confirm.

The helper uses the same context-menu **Rename** action you would use manually.
It makes no `fetch`, `XMLHttpRequest`, or Onshape REST calls. Before changing
anything it verifies that Onshape's displayed mate count (nine in the pictured
leg) equals the loaded rows and that every one ends in `RF`. It then verifies
each new name and stops at the first failure, reporting how many completed. It
is safe to rerun after correcting the visible problem.

### Right-side copies

If front and rear use the same parts, no internal changes are required. Keep
the template parts and position each inserted leg with its root Mate connector.

### Left-side copies

For each genuinely handed part in `Leg-LF-export` and `Leg-LR-export`:

1. Create or select the proper left-handed mirrored part/configuration.
2. Use **Replace instance** on the corresponding right-handed instance.
3. Let Onshape reapply the existing mates.
4. Resolve any failed reference before proceeding. Explicit Part Studio Mate
   connectors make replacement substantially more robust than implicit face
   selections.

Do not replace geometrically symmetric parts merely because they appear on the
opposite side. Reuse them and let the mounting Mate connector orient them.

## Mirror the limits correctly

Duplicating a tab preserves the original numeric limits. After replacing and
orienting the left-handed parts, inspect the positive direction of every free
degree of freedom.

For any degree of freedom whose positive direction is reversed, convert:

```text
[minimum, maximum] -> [-maximum, -minimum]
```

Examples:

```text
Revolute:    [-20 deg, 45 deg] -> [-45 deg, 20 deg]
Cylindrical axial: [-3 mm, 8 mm] -> [-8 mm, 3 mm]
```

A cylindrical mate can have both rotation and axial translation. Reverse each
interval independently only when that degree of freedom's displayed positive
direction is reversed. Do not blindly reverse every cylindrical limit.

Where useful, define shared variables and use expressions in the left mates:

```text
minimum = -#rightMaximum
maximum = -#rightMinimum
```

This keeps later mechanical-limit changes paired without an API or script.

## Build the dog and flatten it for Publisher

1. Open the final dog Assembly.
2. Insert each of the four unique `Leg-*-export` Assembly tabs once.
3. Add one external Mate from each leg's root Mate connector to the matching
   body Mate connector. The internal nine-mate mechanisms remain flexible.
4. Drag or animate each leg independently. Moving one leg must not move any
   other leg.
5. Create an Onshape version before flattening. Dissolve is recoverable through
   history, but the four export leg tabs are intentionally disposable.
6. In the dog Assembly's Instances list, right-click each inserted leg and
   choose **Dissolve subassembly**. Do this once for RF, RR, LF, and LR.
7. Confirm all component instances and mate features now appear directly in the
   top-level dog Assembly. Each dissolved source tab becomes empty; this is why
   the workflow uses four unique export copies instead of four instances of one
   shared tab.

The resulting top-level assembly contains independent native mates and no
Assembly Mirror motion dependency or nested leg subassembly.

## Export gate

Before opening Omniverse Publisher, verify:

- all four legs move independently;
- all top-level mate names end in the correct unique corner suffix (`RF`, `RR`,
  `LF`, or `LR`);
- every mate has a blue/white good state and there are no red mate indicators;
- revolute and cylindrical axes have the intended hardware sign;
- mirrored angular and axial limits stop at the intended physical positions;
- the body is the single intended root and the linkage is not accidentally
  fixed to the assembly origin;
- Publisher Model preparation lists the expected joints after the four legs
  have been dissolved.

The pictured mechanism contains closed linkages. Independent Onshape mates do
not guarantee that a downstream reduced-coordinate physics articulation can
represent every loop. The exported USD must still be checked for its joint
graph and loaded in Isaac Sim before training.

## Faster same-assembly alternative

Onshape also supports selecting already-mated entities and choosing **Copy
items**, then pasting them. Onshape duplicates the selected entities, Mate
connectors, and Mates in one operation. This is useful when all leg parts
already live at the top level.

For this robot, duplicated Assembly tabs are preferred because they provide a
clean place to replace handed parts and correct left-side limits before the
four finished mechanisms are dissolved into the export assembly.

## Offline four-leg USD repair

When the Publisher ZIP already exists, `generate_four_leg_usd.py` can build the
four-corner asset without Onshape or its API:

```powershell
python .\onshape_mate_mirror\generate_four_leg_usd.py `
  <repaired-one-leg-directory> <four-leg-output-directory> `
  --zip <four-leg-output.zip>
```

It places the Main Hub joints at ±90 mm across X and ±94.111705 mm along Y,
creates RF/RR/LF/LR names, mirrors the RR/LF limit signs, authors 28
articulation-tree revolutes plus eight independent loop closures, and removes
the source loop-anchor preload. It also emits:

- `four-leg-visual.gltf`, a geometry-only file that Onshape can translate for
  visual checking;
- `verify-four-leg.html`, a standalone interactive four-view verifier;
- `VERIFICATION.md` and `robot.json`, containing the exact topology and
  dimension checks.

The glTF is not a substitute for the USD: it cannot carry the PhysX drive,
articulation, joint-limit, or loop-closure schemas.
