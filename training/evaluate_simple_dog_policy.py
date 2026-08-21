"""Parse deterministic V2 rollout metrics and enforce promotion gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED = {
    "core": ("straight", "left_curve", "right_curve", "fast", "slow"),
    "robust": ("straight", "left_curve", "right_curve", "fast", "slow"),
    "goal": (
        "stand", "forward", "reverse", "strafe_left", "strafe_right",
        "turn_left", "turn_right", "diagonal_left",
        "diagonal_right", "diagonal_reverse_left",
        "diagonal_reverse_right", "curve_left", "curve_right", "stop",
    ),
}


def parse_value(value: str):
    if "," in value:
        return [float(item) for item in value.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_segments(path: Path) -> dict[str, dict]:
    segments: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("EVAL_SEGMENT "):
            continue
        record = {}
        for token in line.split()[1:]:
            key, separator, value = token.partition("=")
            if separator:
                record[key] = parse_value(value)
        name = record.get("name")
        if isinstance(name, str):
            segments[name] = record
    return segments


def add_failure(failures: list[str], name: str, message: str) -> None:
    failures.append(f"{name}: {message}")


def check_common(name: str, record: dict, failures: list[str]) -> None:
    if int(record["resets"]) != 0:
        add_failure(failures, name, f"reset/fall count is {record['resets']}")
    if float(record["min_height"]) < 0.10:
        add_failure(failures, name, f"minimum body height is {record['min_height']:.3f} m")


def check_planar_motion(name: str, record: dict, failures: list[str]) -> None:
    command_forward = float(record["command_forward"])
    command_lateral = float(record["command_lateral"])
    mean_forward = float(record["mean_body_forward"])
    mean_lateral = float(record["mean_body_lateral"])
    command_speed = math.hypot(command_forward, command_lateral)
    actual_speed = math.hypot(mean_forward, mean_lateral)
    if actual_speed < 0.65 * command_speed:
        add_failure(
            failures,
            name,
            f"planar speed is {actual_speed:.3f} for {command_speed:.3f} m/s",
        )
    if actual_speed > max(1.40 * command_speed, command_speed + 0.06):
        add_failure(
            failures,
            name,
            f"planar overspeed is {actual_speed:.3f} for {command_speed:.3f} m/s",
        )
    for axis, actual, command in (
        ("forward", mean_forward, command_forward),
        ("lateral", mean_lateral, command_lateral),
    ):
        tolerance = max(0.07, 0.35 * abs(command))
        if abs(actual - command) > tolerance:
            add_failure(
                failures, name,
                f"{axis} tracking is {actual:.3f} for {command:.3f} m/s",
            )
    command_yaw = float(record["command_yaw"])
    mean_yaw = float(record["mean_yaw_rate"])
    if abs(mean_yaw - command_yaw) > 0.12:
        add_failure(
            failures,
            name,
            f"yaw-rate tracking is {mean_yaw:.3f} for {command_yaw:.3f} rad/s",
        )
    expected_heading = command_yaw * int(record["steps"]) * 0.02
    heading_error = math.remainder(
        float(record["heading_delta"]) - expected_heading, 2.0 * math.pi
    )
    if abs(heading_error) > 0.45:
        add_failure(
            failures,
            name,
            f"heading error is {heading_error:.3f} rad",
        )
    if abs(command_yaw) < 0.05:
        duration = int(record["steps"]) * 0.02
        displacement = (
            float(record["forward_displacement"]),
            float(record["lateral_displacement"]),
        )
        direction = (
            command_forward / command_speed,
            command_lateral / command_speed,
        )
        progress = displacement[0] * direction[0] + displacement[1] * direction[1]
        cross_track = -displacement[0] * direction[1] + displacement[1] * direction[0]
        if progress < 0.55 * command_speed * duration:
            add_failure(
                failures, name,
                f"signed displacement progress is {progress:.3f} m",
            )
        if abs(cross_track) > 0.35:
            add_failure(
                failures, name,
                f"cross-track displacement is {cross_track:.3f} m",
            )
    swing = record["swing_fraction_frflbrbl"]
    landings = record["landings_frflbrbl"]
    for index, label in enumerate(("FR", "FL", "BR", "BL")):
        if not 0.03 <= float(swing[index]) <= 0.95:
            add_failure(
                failures, name, f"{label} swing fraction is {float(swing[index]):.3f}"
            )
        if int(landings[index]) < 1:
            add_failure(failures, name, f"{label} did not land")


def check_stationary(name: str, record: dict, failures: list[str]) -> None:
    if abs(float(record["mean_body_forward"])) > 0.06:
        add_failure(
            failures,
            name,
            f"mean forward speed while stopped is {record['mean_body_forward']:.3f} m/s",
        )
    if float(record["mean_abs_body_lateral"]) > 0.06:
        add_failure(
            failures,
            name,
            f"mean lateral speed while stopped is {record['mean_abs_body_lateral']:.3f} m/s",
        )
    if abs(float(record["mean_yaw_rate"])) > 0.08:
        add_failure(
            failures,
            name,
            f"mean yaw rate while stopped is {record['mean_yaw_rate']:.3f} rad/s",
        )
    # Permit a brief settling step and millimetre-scale contact gaps caused by
    # a non-coplanar rough tile, but reject a policy that parks a foot visibly
    # aloft. Contact duty alone cannot make that distinction on uneven ground;
    # the base-relative swing clearance can.
    parked_clearance = float(record.get("mean_swing_foot_clearance", 0.0))
    for index, label in enumerate(("FR", "FL", "BR", "BL")):
        swing_fraction = float(record["swing_fraction_frflbrbl"][index])
        if swing_fraction > 0.20 and parked_clearance > 0.008:
            add_failure(
                failures,
                name,
                f"{label} is parked aloft for {swing_fraction:.3f} of the stop "
                f"with {parked_clearance:.3f} m mean clearance",
            )


def check_turn(name: str, record: dict, failures: list[str]) -> None:
    if abs(float(record["mean_body_forward"])) > 0.08:
        add_failure(
            failures,
            name,
            f"translation during turn is {record['mean_body_forward']:.3f} m/s",
        )
    if float(record["mean_abs_body_lateral"]) > 0.08:
        add_failure(
            failures,
            name,
            f"lateral motion during turn is {record['mean_abs_body_lateral']:.3f} m/s",
        )
    command_yaw = float(record["command_yaw"])
    mean_yaw = float(record["mean_yaw_rate"])
    if abs(mean_yaw - command_yaw) > 0.12:
        add_failure(
            failures,
            name,
            f"yaw-rate tracking is {mean_yaw:.3f} for {command_yaw:.3f} rad/s",
        )
    expected_heading = command_yaw * int(record["steps"]) * 0.02
    heading_error = math.remainder(
        float(record["heading_delta"]) - expected_heading, 2.0 * math.pi
    )
    if abs(heading_error) > 0.45:
        add_failure(failures, name, f"heading error is {heading_error:.3f} rad")
    swing = record["swing_fraction_frflbrbl"]
    landings = record["landings_frflbrbl"]
    for index, label in enumerate(("FR", "FL", "BR", "BL")):
        if not 0.03 <= float(swing[index]) <= 0.95:
            add_failure(
                failures, name, f"{label} swing fraction is {float(swing[index]):.3f}"
            )
        if int(landings[index]) < 1:
            add_failure(failures, name, f"{label} did not land")


def check_gait_quality(
    name: str,
    record: dict,
    failures: list[str],
    *,
    vertical_speed_limit: float,
) -> None:
    """Reject motion that passes commands by sliding or violent action jumps."""
    # This is mean COM speed per stance foot, not the previous sum across all
    # contacting feet. A per-foot metric is comparable across trot/duty phases.
    if float(record["mean_foot_slip"]) > 0.30:
        add_failure(
            failures, name,
            f"mean foot slip is {record['mean_foot_slip']:.3f} m/s",
        )
    if float(record["mean_action_rate"]) > 2.50:
        add_failure(
            failures, name,
            f"mean squared action change is {record['mean_action_rate']:.3f}",
        )
    if float(record["max_action_step"]) > 0.42:
        add_failure(
            failures, name,
            f"maximum normalized action step is {record['max_action_step']:.3f}",
        )
    mean_hip_abduction = float(record.get("mean_abs_hip_abduction", 0.0))
    if mean_hip_abduction > 0.14:
        add_failure(
            failures, name,
            f"mean hip abduction is {mean_hip_abduction:.3f} rad",
        )
    max_hip_abduction = float(record.get("max_abs_hip_abduction", 0.0))
    if max_hip_abduction > 0.24:
        add_failure(
            failures, name,
            f"maximum hip abduction is {max_hip_abduction:.3f} rad",
        )
    if float(record["mean_abs_vertical_speed"]) > vertical_speed_limit:
        add_failure(
            failures, name,
            "mean vertical speed is "
            f"{record['mean_abs_vertical_speed']:.3f} m/s "
            f"(limit {vertical_speed_limit:.3f})",
        )
    swing = [float(value) for value in record["swing_fraction_frflbrbl"]]
    command_speed = math.hypot(
        float(record["command_forward"]), float(record["command_lateral"])
    )
    swing_limit = 0.20 if command_speed > 0.05 else 0.25
    if max(swing) - min(swing) > swing_limit:
        add_failure(
            failures, name,
            "four-foot swing fractions are unbalanced "
            f"({','.join(f'{value:.3f}' for value in swing)})",
        )
    landings = [int(value) for value in record["landings_frflbrbl"]]
    if max(landings) - min(landings) > 3:
        add_failure(
            failures, name,
            "four-foot landing counts are unbalanced "
            f"({','.join(str(value) for value in landings)})",
        )


def evaluate(
    stage: str, segments: dict[str, dict], *, require_gait_quality: bool = False
) -> dict:
    expected = EXPECTED[stage]
    failures: list[str] = []
    missing = [name for name in expected if name not in segments]
    if missing:
        failures.append(f"missing evaluation segments: {', '.join(missing)}")
        return {"stage": stage, "passed": False, "failures": failures, "segments": segments}

    for name in expected:
        record = segments[name]
        check_common(name, record, failures)
        command_forward = float(record["command_forward"])
        command_lateral = float(record["command_lateral"])
        command_yaw = float(record["command_yaw"])
        command_speed = math.hypot(command_forward, command_lateral)
        if command_speed > 0.05:
            check_planar_motion(name, record, failures)
        elif abs(command_yaw) > 0.05:
            check_turn(name, record, failures)
        else:
            check_stationary(name, record, failures)
        if require_gait_quality and (command_speed > 0.05 or abs(command_yaw) > 0.05):
            # Robust evaluation injects repeated planar velocity impulses and
            # begins from an 8-degree tilt. Permit the bounded recovery
            # transient while leaving the unperturbed Core/Goal gait limit at
            # 0.100 m/s.
            check_gait_quality(
                name,
                record,
                failures,
                vertical_speed_limit=0.115 if stage == "robust" else 0.100,
            )
    return {
        "stage": stage,
        "passed": not failures,
        "failures": failures,
        "segments": {name: segments[name] for name in expected},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("console_log", type=Path)
    parser.add_argument("--stage", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gait-quality", action="store_true")
    args = parser.parse_args()

    result = evaluate(
        args.stage,
        read_segments(args.console_log),
        require_gait_quality=args.require_gait_quality,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
