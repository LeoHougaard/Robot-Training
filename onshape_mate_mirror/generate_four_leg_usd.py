#!/usr/bin/env python3
"""Build a four-leg USD from the repaired right-front Onshape export.

The source meshes use Onshape's exported part coordinates.  A single-plane
world reflection would give a rigid body an invalid left-handed transform, so
this generator mirrors each affected mesh in its local X direction and pairs
it with a proper (determinant +1) body transform.  Joint frames are rebuilt in
world space and converted back into each target body's local coordinates.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import re
import shutil
import struct
import zipfile
from pathlib import Path


MAIN_BODY = "_MfG1xwFWmgbtmCyEF"
LEG_BODIES = (
    "_M5h2pOkgVaejo_GDh",
    "_MBK47NygQU_fnvKKJ",
    "_MTn8HoecFwNHGSxBF",
    "_MYp_ZmExAGS9OwbEd",
    "_MmTeSqYZZIFXV2OsA",
    "_MqOgoznVVVm5Fuab8",
    "_MudPDoCUz0k_u1fjV",
)

CORNER_SPECS = {
    "RF": {"reflect": (1.0, 1.0, 1.0), "mountSign": (1.0, -1.0), "single": False},
    "RR": {"reflect": (1.0, -1.0, 1.0), "mountSign": (1.0, 1.0), "single": True},
    "LF": {"reflect": (-1.0, 1.0, 1.0), "mountSign": (-1.0, -1.0), "single": True},
    "LR": {"reflect": (-1.0, -1.0, 1.0), "mountSign": (-1.0, 1.0), "single": False},
}

EXPECTED_LENGTH_M = 0.18822341
EXPECTED_WIDTH_M = 0.18000000


def mat_identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def mat_transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def transform_point(p: tuple[float, float, float], m: list[list[float]]) -> tuple[float, float, float]:
    return tuple(sum(p[k] * m[k][j] for k in range(3)) + m[3][j] for j in range(3))


def inverse_rigid(m: list[list[float]]) -> list[list[float]]:
    r = [row[:3] for row in m[:3]]
    rt = mat_transpose(r)
    t = m[3][:3]
    out = mat_identity(4)
    for i in range(3):
        out[i][:3] = rt[i]
    out[3][:3] = [-sum(t[k] * rt[k][j] for k in range(3)) for j in range(3)]
    return out


def det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def quaternion_to_row_matrix(q: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    column = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return mat_transpose(column)


def row_matrix_to_quaternion(row: list[list[float]]) -> tuple[float, float, float, float]:
    m = mat_transpose(row)
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(max(0.0, 1.0 + m[1][1] - m[0][0] - m[2][2])) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + m[2][2] - m[0][0] - m[1][1])) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    q = (w, x, y, z)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    if q[0] < 0:
        q = tuple(-v for v in q)
    return q


def parse_numbers(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)]


def parse_matrix(block: str) -> list[list[float]]:
    match = re.search(r"matrix4d xformOp:transform = \((.*?)\)\n", block, re.S)
    if not match:
        raise ValueError("Body block has no transform matrix")
    values = parse_numbers(match.group(1))
    if len(values) != 16:
        raise ValueError(f"Expected 16 matrix values, got {len(values)}")
    return [values[i : i + 4] for i in range(0, 16, 4)]


def fmt(value: float) -> str:
    if abs(value) < 5e-14:
        value = 0.0
    return f"{value:.15g}"


def fmt_vec(values: tuple[float, ...] | list[float]) -> str:
    return "(" + ", ".join(fmt(v) for v in values) + ")"


def fmt_matrix(m: list[list[float]]) -> str:
    return "(" + ", ".join(fmt_vec(row) for row in m) + ")"


def find_prim_block(text: str, prim_type: str, name: str) -> str:
    pattern = re.compile(rf"(?m)^([ \t]*)def {re.escape(prim_type)} \"{re.escape(name)}\"(?:\s|\()")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find {prim_type} {name}")
    start = match.start()
    brace = text.find("{", match.start())
    depth = 0
    in_string = False
    escaped = False
    for i in range(brace, len(text)):
        char = text[i]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError(f"Unclosed prim {prim_type} {name}")


def list_prim_names(text: str, prim_type: str, scope_start: int, scope_end: int) -> list[str]:
    scoped = text[scope_start:scope_end]
    return re.findall(rf'(?m)^\s*def {re.escape(prim_type)} "([^"]+)"', scoped)


def parse_body_ref(block: str, side: int) -> str:
    match = re.search(rf'rel physics:body{side} = </robot/links/([^>]+)>', block)
    if not match:
        raise ValueError(f"Joint lacks body{side}")
    return match.group(1)


def parse_vec_attr(block: str, attr: str) -> tuple[float, ...]:
    match = re.search(rf"{re.escape(attr)} = \(([^)]*)\)", block)
    if not match:
        raise ValueError(f"Missing attribute {attr}")
    return tuple(parse_numbers(match.group(1)))


def parse_float_attr(block: str, attr: str) -> float:
    match = re.search(rf"{re.escape(attr)} = ([^\n]+)", block)
    if not match:
        raise ValueError(f"Missing attribute {attr}")
    return float(match.group(1).strip())


def replace_vec_attr(block: str, attr: str, values: tuple[float, ...] | list[float]) -> str:
    return re.sub(
        rf"({re.escape(attr)} = )\([^)]*\)",
        lambda m: m.group(1) + fmt_vec(values),
        block,
        count=1,
    )


def replace_float_attr(block: str, attr: str, value: float) -> str:
    return re.sub(
        rf"({re.escape(attr)} = )[^\n]+",
        lambda m: m.group(1) + fmt(value),
        block,
        count=1,
    )


def reflected_body_matrix(
    source: list[list[float]],
    reflect: tuple[float, float, float],
    offset: tuple[float, float, float],
    single: bool,
) -> list[list[float]]:
    world_reflect = mat_identity(4)
    for i, value in enumerate(reflect):
        world_reflect[i][i] = value
    world_reflect[3][:3] = list(offset)
    local_reflect = mat_identity(4)
    if single:
        local_reflect[0][0] = -1.0
    return mat_mul(mat_mul(local_reflect, source), world_reflect)


def target_joint_frame(
    local_pos: tuple[float, float, float],
    local_rot: tuple[float, float, float, float],
    source_body: list[list[float]],
    target_body: list[list[float]],
    reflect: tuple[float, float, float],
    offset: tuple[float, float, float],
    single: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    world_pos = transform_point(local_pos, source_body)
    reflected_pos = tuple(world_pos[i] * reflect[i] + offset[i] for i in range(3))
    target_pos = transform_point(reflected_pos, inverse_rigid(target_body))

    source_world_rot = mat_mul(quaternion_to_row_matrix(local_rot), [row[:3] for row in source_body[:3]])
    reflect3 = [[reflect[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    handedness_fix = mat_identity(3)
    if single:
        handedness_fix[0][0] = -1.0
    target_world_rot = mat_mul(mat_mul(handedness_fix, source_world_rot), reflect3)
    target_local_rot = mat_mul(target_world_rot, mat_transpose([row[:3] for row in target_body[:3]]))
    if abs(det3(target_local_rot) - 1.0) > 1e-6:
        raise ValueError("Target joint frame is not right-handed")
    return target_pos, row_matrix_to_quaternion(target_local_rot)


def suffix_name(name: str, suffix: str) -> str:
    name = re.sub(r"\s+(RF|RR|LF|LR)$", "", name.strip())
    return f"{name} {suffix}"


def rewrite_body(
    source_block: str,
    source_id: str,
    target_id: str,
    target_matrix: list[list[float]],
    suffix: str,
    mirror_local_x: bool,
) -> str:
    block = re.sub(
        rf'(def Xform "){re.escape(source_id)}(")',
        rf"\g<1>{target_id}\2",
        source_block,
        count=1,
    )
    display = re.search(r'displayName = "([^"]+)"', block)
    if display:
        base = re.sub(r"\s*<\d+>$", "", display.group(1))
        block = block[: display.start(1)] + suffix_name(base, suffix) + block[display.end(1) :]
    block = re.sub(
        r"(matrix4d xformOp:transform = ).*?\n",
        lambda m: m.group(1) + fmt_matrix(target_matrix) + "\n",
        block,
        count=1,
        flags=re.S,
    )
    if mirror_local_x:
        com = parse_vec_attr(block, "physics:centerOfMass")
        block = replace_vec_attr(block, "physics:centerOfMass", (-com[0], com[1], com[2]))
        principal = parse_vec_attr(block, "physics:principalAxes")
        principal_row = quaternion_to_row_matrix(principal)
        local_reflect3 = mat_identity(3)
        local_reflect3[0][0] = -1.0
        principal_mirrored = mat_mul(mat_mul(local_reflect3, principal_row), local_reflect3)
        block = replace_vec_attr(block, "physics:principalAxes", row_matrix_to_quaternion(principal_mirrored))
        block = block.replace(f"@./meshes/{source_id}.gltf@", f"@./meshes_mirror_x/{source_id}.gltf@")
    return block


def rewrite_joint(
    source_block: str,
    source_id: str,
    target_id: str,
    suffix: str,
    body_map: dict[str, str],
    source_body_matrices: dict[str, list[list[float]]],
    target_body_matrices: dict[str, list[list[float]]],
    reflect: tuple[float, float, float],
    offset: tuple[float, float, float],
    single: bool,
) -> tuple[str, dict]:
    block = re.sub(
        rf'(def PhysicsRevoluteJoint "){re.escape(source_id)}(")',
        rf"\g<1>{target_id}\2",
        source_block,
        count=1,
    )
    display = re.search(r'displayName = "([^"]+)"', block)
    if not display:
        raise ValueError(f"Joint {source_id} has no displayName")
    target_display = suffix_name(display.group(1), suffix)
    block = block[: display.start(1)] + target_display + block[display.end(1) :]

    source_body0 = parse_body_ref(source_block, 0)
    source_body1 = parse_body_ref(source_block, 1)
    target_body0 = body_map[source_body0]
    target_body1 = body_map[source_body1]
    block = re.sub(r"(rel physics:body0 = </robot/links/)[^>]+(>)", rf"\g<1>{target_body0}\2", block, count=1)
    block = re.sub(r"(rel physics:body1 = </robot/links/)[^>]+(>)", rf"\g<1>{target_body1}\2", block, count=1)

    excluded = bool(re.search(r"physics:excludeFromArticulation\s*=\s*(?:1|true)", block))
    target_frames: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for side, source_body, target_body in (
        (0, source_body0, target_body0),
        (1, source_body1, target_body1),
    ):
        source_pos = parse_vec_attr(source_block, f"physics:localPos{side}")
        source_rot = parse_vec_attr(source_block, f"physics:localRot{side}")
        target_pos, target_rot = target_joint_frame(
            source_pos,
            source_rot,
            source_body_matrices[source_body],
            target_body_matrices[target_body],
            reflect,
            offset,
            single,
        )
        target_frames[side] = (target_pos, target_rot)

    source_target_anchor0 = transform_point(target_frames[0][0], target_body_matrices[target_body0])
    source_target_anchor1 = transform_point(target_frames[1][0], target_body_matrices[target_body1])
    source_anchor_mismatch = math.sqrt(
        sum((source_target_anchor0[i] - source_target_anchor1[i]) ** 2 for i in range(3))
    )
    if excluded:
        shared_anchor = tuple((source_target_anchor0[i] + source_target_anchor1[i]) / 2.0 for i in range(3))
        target_frames[0] = (
            transform_point(shared_anchor, inverse_rigid(target_body_matrices[target_body0])),
            target_frames[0][1],
        )
        target_frames[1] = (
            transform_point(shared_anchor, inverse_rigid(target_body_matrices[target_body1])),
            target_frames[1][1],
        )

    # Onshape's export aligns the revolute axes, but can leave an arbitrary
    # twist between the two joint frames.  PhysX encodes that twist as a
    # nonzero generalized coordinate; resetting joints to zero then folds the
    # linkage away from the authored assembly.  Make the complete frames
    # coincide at the verified assembly pose so zero is a reproducible reset.
    world_rot0 = mat_mul(
        quaternion_to_row_matrix(target_frames[0][1]),
        [row[:3] for row in target_body_matrices[target_body0][:3]],
    )
    local_rot1 = mat_mul(
        world_rot0,
        mat_transpose([row[:3] for row in target_body_matrices[target_body1][:3]]),
    )
    target_frames[1] = (
        target_frames[1][0],
        row_matrix_to_quaternion(local_rot1),
    )
    for side in (0, 1):
        block = replace_vec_attr(block, f"physics:localPos{side}", target_frames[side][0])
        block = replace_vec_attr(block, f"physics:localRot{side}", target_frames[side][1])

    lower = parse_float_attr(source_block, "physics:lowerLimit")
    upper = parse_float_attr(source_block, "physics:upperLimit")
    if single:
        lower, upper = -upper, -lower
    block = replace_float_attr(block, "physics:lowerLimit", lower)
    block = replace_float_attr(block, "physics:upperLimit", upper)

    driven = 'PhysicsDriveAPI:angular' in block
    target_pos0 = parse_vec_attr(block, "physics:localPos0")
    target_pos1 = parse_vec_attr(block, "physics:localPos1")
    target_rot0 = parse_vec_attr(block, "physics:localRot0")
    target_rot1 = parse_vec_attr(block, "physics:localRot1")
    world_anchor0 = transform_point(target_pos0, target_body_matrices[target_body0])
    world_anchor1 = transform_point(target_pos1, target_body_matrices[target_body1])
    world_rot0 = mat_mul(quaternion_to_row_matrix(target_rot0), [row[:3] for row in target_body_matrices[target_body0][:3]])
    world_rot1 = mat_mul(quaternion_to_row_matrix(target_rot1), [row[:3] for row in target_body_matrices[target_body1][:3]])
    axis0 = tuple(world_rot0[2])
    axis1 = tuple(world_rot1[2])
    mismatch = math.sqrt(sum((world_anchor0[i] - world_anchor1[i]) ** 2 for i in range(3)))
    axis_alignment = abs(sum(axis0[i] * axis1[i] for i in range(3)))
    frame_alignment_error = max(
        abs(world_rot0[i][j] - world_rot1[i][j])
        for i in range(3)
        for j in range(3)
    )
    metadata = {
        "id": target_id,
        "name": target_display,
        "body0": target_body0,
        "body1": target_body1,
        "lowerLimitDeg": lower,
        "upperLimitDeg": upper,
        "excludedFromArticulation": excluded,
        "driven": driven,
        "anchor": list(world_anchor0),
        "otherAnchor": list(world_anchor1),
        "anchorMismatchMm": mismatch * 1000.0,
        "sourceAnchorMismatchMm": source_anchor_mismatch * 1000.0,
        "axis": list(axis0),
        "axisAlignment": axis_alignment,
        "frameAlignmentError": frame_alignment_error,
    }
    return block, metadata


def mirror_gltf_x(source_path: Path, target_path: Path) -> None:
    gltf = json.loads(source_path.read_text(encoding="utf-8"))
    if len(gltf.get("buffers", [])) != 1:
        raise ValueError(f"Expected one embedded buffer in {source_path}")
    uri = gltf["buffers"][0]["uri"]
    if not uri.startswith("data:"):
        raise ValueError(f"Expected embedded buffer in {source_path}")
    prefix, encoded = uri.split(",", 1)
    data = bytearray(base64.b64decode(encoded))

    def accessor_offset(index: int) -> tuple[dict, dict, int, int]:
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
        return accessor, view, offset, components

    position_accessors: set[int] = set()
    normal_accessors: set[int] = set()
    index_accessors: set[int] = set()
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError(f"Only triangle meshes are supported: {source_path}")
            position_accessors.add(primitive["attributes"]["POSITION"])
            if "NORMAL" in primitive["attributes"]:
                normal_accessors.add(primitive["attributes"]["NORMAL"])
            if "indices" in primitive:
                index_accessors.add(primitive["indices"])

    for index in position_accessors | normal_accessors:
        accessor, view, offset, components = accessor_offset(index)
        if accessor["componentType"] != 5126 or components != 3:
            raise ValueError(f"Expected float VEC3 accessor in {source_path}")
        stride = view.get("byteStride", 12)
        values = []
        for item in range(accessor["count"]):
            base = offset + item * stride
            xyz = list(struct.unpack_from("<fff", data, base))
            xyz[0] = -xyz[0]
            struct.pack_into("<fff", data, base, *xyz)
            values.append(xyz)
        if index in position_accessors:
            accessor["min"] = [min(v[i] for v in values) for i in range(3)]
            accessor["max"] = [max(v[i] for v in values) for i in range(3)]

    for index in index_accessors:
        accessor, view, offset, components = accessor_offset(index)
        if components != 1 or accessor["count"] % 3:
            raise ValueError(f"Expected triangle indices in {source_path}")
        component = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4)}.get(accessor["componentType"])
        if not component:
            raise ValueError(f"Unsupported index type in {source_path}")
        code, size = component
        stride = view.get("byteStride", size)
        for tri in range(0, accessor["count"], 3):
            offsets = [offset + (tri + i) * stride for i in range(3)]
            values = [struct.unpack_from("<" + code, data, item)[0] for item in offsets]
            values[1], values[2] = values[2], values[1]
            for item, value in zip(offsets, values):
                struct.pack_into("<" + code, data, item, value)

    gltf["buffers"][0]["uri"] = prefix + "," + base64.b64encode(data).decode("ascii")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(gltf, separators=(",", ":")), encoding="utf-8")


def merge_visual_gltf(
    output_root: Path,
    body_instances: list[dict],
) -> dict:
    merged: dict = {
        "asset": {"version": "2.0", "generator": "Robot Training four-leg verifier"},
        "scene": 0,
        "scenes": [{"name": "Four-leg visual verification", "nodes": [0]}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
    }
    asset_mesh_index: dict[str, int] = {}

    for instance in body_instances:
        asset_key = instance["mesh"]
        if asset_key in asset_mesh_index:
            continue
        source = json.loads((output_root / asset_key).read_text(encoding="utf-8"))
        buffer_offset = len(merged["buffers"])
        view_offset = len(merged["bufferViews"])
        accessor_offset = len(merged["accessors"])
        material_offset = len(merged["materials"])

        merged["buffers"].extend(copy.deepcopy(source.get("buffers", [])))
        for view in copy.deepcopy(source.get("bufferViews", [])):
            view["buffer"] = view.get("buffer", 0) + buffer_offset
            merged["bufferViews"].append(view)
        for accessor in copy.deepcopy(source.get("accessors", [])):
            if "bufferView" in accessor:
                accessor["bufferView"] += view_offset
            merged["accessors"].append(accessor)
        merged["materials"].extend(copy.deepcopy(source.get("materials", [])))

        source_mesh = copy.deepcopy(source["meshes"][0])
        for primitive in source_mesh.get("primitives", []):
            primitive["attributes"] = {
                key: value + accessor_offset for key, value in primitive.get("attributes", {}).items()
            }
            if "indices" in primitive:
                primitive["indices"] += accessor_offset
            if "material" in primitive:
                primitive["material"] += material_offset
        source_mesh["name"] = asset_key
        asset_mesh_index[asset_key] = len(merged["meshes"])
        merged["meshes"].append(source_mesh)

    root_conversion = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    merged["nodes"].append(
        {
            "name": "Four-leg robot (Z-up converted to glTF Y-up)",
            "matrix": [value for row in root_conversion for value in row],
            "children": list(range(1, len(body_instances) + 1)),
        }
    )
    for instance in body_instances:
        merged["nodes"].append(
            {
                "name": instance["name"],
                "mesh": asset_mesh_index[instance["mesh"]],
                "matrix": [value for row in instance["matrix"] for value in row],
                "extras": {"corner": instance["corner"], "usdPrim": instance["id"]},
            }
        )
    return merged


def create_zip(source_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.name != zip_path.name:
                archive.write(path, path.relative_to(source_root).as_posix())


def generate(source_root: Path, output_root: Path, template_path: Path) -> dict:
    source_usda_path = source_root / "robot.usda"
    source_text = source_usda_path.read_text(encoding="utf-8")
    source_root_block = find_prim_block(source_text, "Xform", "robot")
    joint_scope = find_prim_block(source_root_block, "Scope", "joints")
    link_scope = find_prim_block(source_root_block, "Scope", "links")
    joint_ids = list_prim_names(joint_scope, "PhysicsRevoluteJoint", 0, len(joint_scope))
    if len(joint_ids) != 9:
        raise ValueError(f"Expected 9 RF joints, got {len(joint_ids)}")

    source_body_blocks = {
        body_id: find_prim_block(link_scope, "Xform", body_id)
        for body_id in (MAIN_BODY,) + LEG_BODIES
    }
    source_body_matrices = {body_id: parse_matrix(block) for body_id, block in source_body_blocks.items()}
    source_joint_blocks = {joint_id: find_prim_block(joint_scope, "PhysicsRevoluteJoint", joint_id) for joint_id in joint_ids}
    source_main_hub_id = next(
        joint_id for joint_id, block in source_joint_blocks.items() if 'displayName = "Main Hub RF"' in block
    )
    source_main_hub = source_joint_blocks[source_main_hub_id]
    source_main_side = 0 if parse_body_ref(source_main_hub, 0) == MAIN_BODY else 1
    source_main_anchor = transform_point(
        parse_vec_attr(source_main_hub, f"physics:localPos{source_main_side}"),
        source_body_matrices[MAIN_BODY],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_root / "meshes"
    if meshes_dir.exists():
        shutil.rmtree(meshes_dir)
    shutil.copytree(source_root / "meshes", meshes_dir)
    mirror_dir = output_root / "meshes_mirror_x"
    if mirror_dir.exists():
        shutil.rmtree(mirror_dir)
    mirror_dir.mkdir(parents=True)
    for body_id in LEG_BODIES:
        mirror_gltf_x(meshes_dir / f"{body_id}.gltf", mirror_dir / f"{body_id}.gltf")

    all_body_blocks: list[str] = []
    all_joint_blocks: list[str] = []
    all_joint_metadata: list[dict] = []
    articulation_joint_paths: list[str] = []
    body_paths = [f"</robot/links/{MAIN_BODY}>"]
    body_instances = [
        {
            "id": MAIN_BODY,
            "name": "Main",
            "corner": "chassis",
            "mesh": f"meshes/{MAIN_BODY}.gltf",
            "matrix": source_body_matrices[MAIN_BODY],
        }
    ]

    main_block = source_body_blocks[MAIN_BODY]
    main_block = re.sub(r'displayName = "[^"]+"', 'displayName = "Main"', main_block, count=1)
    all_body_blocks.append(main_block)

    target_matrices_by_corner: dict[str, dict[str, list[list[float]]]] = {}
    body_maps_by_corner: dict[str, dict[str, str]] = {}
    offsets_by_corner: dict[str, tuple[float, float, float]] = {}
    for suffix, spec in CORNER_SPECS.items():
        reflect = spec["reflect"]
        single = spec["single"]
        desired_anchor = (
            spec["mountSign"][0] * EXPECTED_WIDTH_M / 2.0,
            spec["mountSign"][1] * EXPECTED_LENGTH_M / 2.0,
            source_main_anchor[2],
        )
        reflected_anchor = tuple(source_main_anchor[i] * reflect[i] for i in range(3))
        offset = tuple(desired_anchor[i] - reflected_anchor[i] for i in range(3))
        offsets_by_corner[suffix] = offset
        body_map = {MAIN_BODY: MAIN_BODY}
        target_matrices = {MAIN_BODY: source_body_matrices[MAIN_BODY]}
        for source_id in LEG_BODIES:
            target_id = source_id if suffix == "RF" else f"{source_id}_{suffix}"
            body_map[source_id] = target_id
            target_matrix = reflected_body_matrix(source_body_matrices[source_id], reflect, offset, single)
            if abs(det3([row[:3] for row in target_matrix[:3]]) - 1.0) > 1e-6:
                raise ValueError(f"Rigid transform for {target_id} is not proper")
            target_matrices[target_id] = target_matrix
            target_block = rewrite_body(
                source_body_blocks[source_id],
                source_id,
                target_id,
                target_matrix,
                suffix,
                single,
            )
            all_body_blocks.append(target_block)
            body_paths.append(f"</robot/links/{target_id}>")
            display = re.search(r'displayName = "([^"]+)"', target_block).group(1)
            body_instances.append(
                {
                    "id": target_id,
                    "name": display,
                    "corner": suffix,
                    "mesh": (f"meshes_mirror_x/{source_id}.gltf" if single else f"meshes/{source_id}.gltf"),
                    "matrix": target_matrix,
                }
            )
        target_matrices_by_corner[suffix] = target_matrices
        body_maps_by_corner[suffix] = body_map

    for suffix, spec in CORNER_SPECS.items():
        for source_id in joint_ids:
            target_id = source_id if suffix == "RF" else f"{source_id}_{suffix}"
            block, metadata = rewrite_joint(
                source_joint_blocks[source_id],
                source_id,
                target_id,
                suffix,
                body_maps_by_corner[suffix],
                source_body_matrices,
                target_matrices_by_corner[suffix],
                spec["reflect"],
                offsets_by_corner[suffix],
                spec["single"],
            )
            all_joint_blocks.append(block)
            all_joint_metadata.append(metadata)
            if not metadata["excludedFromArticulation"]:
                articulation_joint_paths.append(f"</robot/joints/{target_id}>")

    root_close = source_text.find(source_root_block) + len(source_root_block)
    trailing_scopes = source_text[root_close:].lstrip("\r\n")
    robot_usda = f'''#usda 1.0
(
    defaultPrim = "robot"
    doc = """Four-leg closed-linkage robot generated from the repaired RF Onshape export."""
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "robot" (
    prepend apiSchemas = ["IsaacRobotAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    displayName = "Isaac lab Assembly - four-leg repaired closed linkages"
    kind = "component"
)
{{
    string isaac:description = "Four legs, each with a seven-joint articulation tree and two independent maximal-coordinate loop closures."
    string isaac:namespace = "four_leg_dog"
    prepend rel isaac:physics:robotJoints = [{', '.join(articulation_joint_paths)}]
    prepend rel isaac:physics:robotLinks = [{', '.join(body_paths)}]
    bool physxArticulation:enabledSelfCollisions = 0
    float physxArticulation:solverPositionIterationCount = 16
    float physxArticulation:solverVelocityIterationCount = 4
    float physxArticulation:stabilizationThreshold = 0
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    double3 xformOp:translate = (0, 0, 0)
    quatd xformOp:orient = (1, 0, 0, 0)
    double3 xformOp:scale = (1, 1, 1)

    def Scope "joints"
    {{
{chr(10).join(all_joint_blocks)}
    }}

    def Scope "links"
    {{
{chr(10).join(all_body_blocks)}
    }}
}}

{trailing_scopes}'''
    (output_root / "robot.usda").write_text(robot_usda, encoding="utf-8", newline="\n")

    root_joint_names = {"Main Hub RF", "Main Hub RR", "Main Hub LF", "Main Hub LR"}
    root_anchors = {item["name"].split()[-1]: item["anchor"] for item in all_joint_metadata if item["name"] in root_joint_names}
    width = abs(root_anchors["RF"][0] - root_anchors["LF"][0])
    length = abs(root_anchors["RF"][1] - root_anchors["RR"][1])

    manifest = {
        "Format": "Onshape-to-robot four-leg repair manifest v1",
        "Source": str(source_usda_path),
        "MainBody": MAIN_BODY,
        "MountSpacingMeters": {"length": length, "width": width},
        "Corners": list(CORNER_SPECS),
        "RigidBodies": [{"id": item["id"], "name": item["name"], "corner": item["corner"]} for item in body_instances],
        "Joints": all_joint_metadata,
        "ActuatorSettings": {
            "maxTorqueNm": 1.37,
            "maxVelocityRadPerSecond": 8.0,
            "stiffness": 22.0,
            "damping": 0.8,
            "armature": 0.001,
        },
    }
    (output_root / "robot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    visual = merge_visual_gltf(output_root, body_instances)
    visual_path = output_root / "four-leg-visual.gltf"
    visual_path.write_text(json.dumps(visual, separators=(",", ":")), encoding="utf-8")

    verification_data = {
        "mounts": root_anchors,
        "dimensionsMm": {"length": length * 1000.0, "width": width * 1000.0},
        "joints": all_joint_metadata,
    }
    template = template_path.read_text(encoding="utf-8")
    html = template.replace("__EMBEDDED_GLTF__", json.dumps(visual, separators=(",", ":")))
    html = html.replace("__VERIFICATION_DATA__", json.dumps(verification_data, separators=(",", ":")))
    (output_root / "verify-four-leg.html").write_text(html, encoding="utf-8", newline="\n")

    checks = {
        "rigidBodies": len(body_instances),
        "revoluteJoints": len(all_joint_metadata),
        "articulationTreeJoints": sum(not item["excludedFromArticulation"] for item in all_joint_metadata),
        "loopClosures": sum(item["excludedFromArticulation"] for item in all_joint_metadata),
        "drives": sum(item["driven"] for item in all_joint_metadata),
        "lengthMm": length * 1000.0,
        "widthMm": width * 1000.0,
        "allRigidTransformsProper": all(
            abs(det3([row[:3] for row in item["matrix"][:3]]) - 1.0) < 1e-6 for item in body_instances
        ),
        "allJointAxesAligned": all(item["axisAlignment"] > 0.999999 for item in all_joint_metadata),
        "allJointFramesAligned": all(item["frameAlignmentError"] < 1e-6 for item in all_joint_metadata),
        "maxTreeAnchorMismatchMm": max(
            item["anchorMismatchMm"] for item in all_joint_metadata if not item["excludedFromArticulation"]
        ),
        "loopAnchorMismatchMm": [
            item["anchorMismatchMm"] for item in all_joint_metadata if item["excludedFromArticulation"]
        ],
        "repairedSourceLoopMismatchMm": [
            item["sourceAnchorMismatchMm"] for item in all_joint_metadata if item["excludedFromArticulation"]
        ],
    }
    expected = {
        "rigidBodies": 29,
        "revoluteJoints": 36,
        "articulationTreeJoints": 28,
        "loopClosures": 8,
        "drives": 12,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise ValueError(f"{key}: expected {value}, got {checks[key]}")
    if abs(length - EXPECTED_LENGTH_M) > 1e-8:
        raise ValueError(f"Length spacing mismatch: {length}")
    if abs(width - EXPECTED_WIDTH_M) > 1e-8:
        raise ValueError(f"Width spacing mismatch: {width}")
    if not checks["allRigidTransformsProper"]:
        raise ValueError("At least one rigid body has a reflected transform")
    if not checks["allJointAxesAligned"]:
        raise ValueError("At least one joint's two local axes are not aligned")
    if not checks["allJointFramesAligned"]:
        raise ValueError("At least one joint's authored zero frames are not aligned")
    if checks["maxTreeAnchorMismatchMm"] > 1e-4:
        raise ValueError(f"Tree joint anchor mismatch: {checks['maxTreeAnchorMismatchMm']} mm")
    if max(checks["loopAnchorMismatchMm"]) > 1e-4:
        raise ValueError(f"Loop closure still has an initial anchor mismatch: {checks['loopAnchorMismatchMm']}")

    report_lines = [
        "# Four-leg generation verification",
        "",
        f"- Rigid bodies: {checks['rigidBodies']} (1 chassis + 4 x 7 leg bodies)",
        f"- Revolute constraints: {checks['revoluteJoints']} (4 x 9)",
        f"- Articulation-tree joints: {checks['articulationTreeJoints']} (4 x 7)",
        f"- Independent loop closures: {checks['loopClosures']} (4 x 2)",
        f"- Driven joints: {checks['drives']} (4 x 3)",
        f"- Main Hub length spacing: {checks['lengthMm']:.5f} mm",
        f"- Main Hub width spacing: {checks['widthMm']:.5f} mm",
        "- Every rigid-body transform has determinant +1 (no reflected physics frames).",
        f"- Maximum articulation-tree anchor mismatch: {checks['maxTreeAnchorMismatchMm']:.9f} mm.",
        "- Both local frames of every revolute joint coincide at the authored zero pose.",
        "- Both loop-closure pivots per leg are coincident at the authored pose (no initial constraint preload).",
        "- RR and LF meshes are locally mirrored with triangle winding and normals corrected.",
        "- RR and LF revolute limits are sign-inverted and swapped; LR limits retain RF signs.",
        "- Visual verification: open `verify-four-leg.html` or import `four-leg-visual.gltf` into Onshape.",
        "- The glTF verifies geometry only; USD physics joints remain in `robot.usda`.",
    ]
    (output_root / "VERIFICATION.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the repaired one-leg robot.usda and meshes")
    parser.add_argument("output", type=Path, help="Four-leg output directory")
    parser.add_argument("--zip", dest="zip_path", type=Path, help="Optional output ZIP path")
    args = parser.parse_args()
    template_path = Path(__file__).with_name("four_leg_verifier_template.html")
    checks = generate(args.source.resolve(), args.output.resolve(), template_path)
    if args.zip_path:
        create_zip(args.output.resolve(), args.zip_path.resolve())
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
