"""Print compact learning diagnostics from one RL-Games TensorBoard event file."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--tail",
        type=int,
        default=10,
        help="Number of recent scalar samples summarized (default: 10).",
    )
    args = parser.parse_args()
    if args.tail < 1:
        parser.error("--tail must be at least 1")

    event_path = args.event_file
    if event_path.is_dir() and (event_path / "summaries").is_dir():
        event_path = event_path / "summaries"
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    requested_tags = (
        "rewards/iter",
        "Episode/Metrics/mean_survival_fraction",
        "Episode/Metrics/mean_velocity_error",
        "Episode/Metrics/mean_world_forward_speed",
        "Episode/Metrics/mean_body_lateral_speed",
        "Episode/Metrics/mean_yaw_rate_error",
        "Episode/Metrics/mean_heading_error",
        "Episode/Metrics/swing_fraction_front_right",
        "Episode/Metrics/swing_fraction_front_left",
        "Episode/Metrics/swing_fraction_back_right",
        "Episode/Metrics/swing_fraction_back_left",
        "Episode/Metrics/landings_front_right",
        "Episode/Metrics/landings_front_left",
        "Episode/Metrics/landings_back_right",
        "Episode/Metrics/landings_back_left",
        "Episode/Metrics/terrain_level",
        "Episode/Metrics/terrain_move_up_fraction",
        "Episode/Metrics/terrain_move_down_fraction",
        "Episode/Episode_Reward/gait",
        "Episode/Episode_Reward/track_body_velocity",
        "Episode/Episode_Reward/track_yaw_rate",
        "Episode/Episode_Reward/feet_air_time",
        "Episode/Episode_Reward/air_time_variance",
        "Episode/Episode_Reward/swing_duty_floor",
        "Episode/Episode_Reward/fall",
        "Episode/Episode_Reward/base_motion",
        "Episode/Episode_Reward/base_orientation",
        "Episode/Episode_Reward/action_smoothness",
        "Episode/Episode_Reward/foot_slip",
        "Episode/Episode_Reward/undesired_contact",
        "Episode/Episode_Reward/locomotion",
        "Episode/Episode_Reward/diagonal_gait",
        "Episode/Episode_Reward/complete_gait_cycle",
        "Episode/Episode_Reward/reference_trot",
        "Episode/Episode_Reward/foot_clearance",
        "Episode/Episode_Reward/uncommanded_motion",
        "Episode/Episode_Reward/prolonged_foot_air",
        "Episode/Episode_Reward/stability",
        "Episode/Episode_Reward/action_rate",
        "Episode/Episode_Reward/hip_abduction",
        "Episode/Episode_Reward/foot_spread",
        "Episode/Episode_Termination/fell",
        "Episode/Episode_Termination/time_out",
    )
    available = set(accumulator.Tags()["scalars"])
    tags = list(requested_tags)
    # RL-Games may prepend a scope to environment metrics. Resolve by suffix
    # so this remains useful across Isaac Lab/RL-Games point releases.
    for requested in requested_tags:
        if requested not in available:
            suffix = requested.split("/", 1)[-1]
            tags.extend(sorted(tag for tag in available if tag.endswith(suffix)))

    seen: set[str] = set()
    records: dict[str, dict[str, float | int] | None] = {}
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        if tag not in available:
            records[tag] = None
            if not args.as_json:
                print(f"{tag}: unavailable")
            continue
        values = accumulator.Scalars(tag)
        first, last = values[0], values[-1]
        recent = values[-args.tail :]
        best = max(values, key=lambda value: value.value)
        records[tag] = {
            "first": first.value,
            "first_step": first.step,
            "last": last.value,
            "last_step": last.step,
            "samples": len(values),
            "tail_samples": len(recent),
            "tail_mean": statistics.fmean(value.value for value in recent),
            "tail_min": min(value.value for value in recent),
            "tail_max": max(value.value for value in recent),
            "max": best.value,
            "max_step": best.step,
        }
        if not args.as_json:
            print(
                f"{tag}: first={first.value:.6g} at {first.step}, "
                f"last={last.value:.6g} at {last.step}, samples={len(values)}"
            )
    if args.as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
    elif not any(tag in available for tag in tags):
        print("Available scalar tags:")
        for tag in sorted(available):
            print(f"  {tag}")


if __name__ == "__main__":
    main()
