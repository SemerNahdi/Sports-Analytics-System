#!/usr/bin/env python3
"""
Mitus AI Sports Analytics System — Entry Point  v6
====================================================
NEW FLOW
--------
When --sports2d is used (recommended):

  STEP 1  Sports2D runs first.
          If --pick is given, Sports2D shows its native on_click picker so
          you select the player once, visually, inside Sports2D's own UI.
          Sports2D produces:
            * annotated video  (Output/sports2d_annotated.mp4)
            * native angle graphs  (Output/Sports2D/*.png)
            * MOT joint-angle file  (Output/Sports2D/*.mot)
            * TRC marker file       (Output/Sports2D/*.trc)

  STEP 2  Custom analysis runs on the SAME player.
          The tracker is seeded automatically from Sports2D's TRC output,
          so no second pick is needed.
          Custom analysis produces:
            * report.txt
            * data_output.json
            * bio_metrics.csv
            * player_markers.trc  (OpenSim)
            * joint_angles.mot    (OpenSim)

NOTE: Output/results/ custom plots are NOT generated anymore.
      Sports2D's own graphs (Output/Sports2D/*.png) are the
      authoritative angle/biomechanics visualisations.

INSTALL:
  pip install ultralytics opencv-python numpy pandas scipy
  pip install sports2d pose2sim
  # ffmpeg on PATH enables container-fix for the annotated video

USAGE:
  python run_analysis.py --video match.mp4 --sports2d --pick
  python run_analysis.py --video match.mp4 --sports2d --pick --s2d-ik --s2d-augment
  python run_analysis.py --video match.mp4               # custom only, no Sports2D
"""

import argparse
import os
import sys

from sports_analytics import SportsAnalyzer, HAS_SPORTS2D


def main():
    parser = argparse.ArgumentParser(
        description="Mitus AI Sports Analytics v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument("--video",    required=True, help="Input video path")
    parser.add_argument("--player",   type=int, default=1, help="Player ID label")
    parser.add_argument("--fps",      type=float, default=None,
                        help="Override detected FPS")
    parser.add_argument("--height",   type=float, default=1.75,
                        help="Player height in metres (default 1.75)")
    parser.add_argument("--mass",     type=float, default=75.0,
                        help="Player mass in kg (default 75.0)")

    # Tracker
    parser.add_argument("--pick",      action="store_true",
                        help="Interactively pick the player. When --sports2d is used, "
                             "this activates Sports2D's native on_click picker.")
    parser.add_argument("--yolo-size", default="m",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size for custom tracker (default: m)")

    # Sports2D
    parser.add_argument("--sports2d",     action="store_true",
                        help="Run Sports2D first (recommended — best graphs & video)")
    parser.add_argument("--s2d-mode",     default="balanced",
                        choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--s2d-ik",       action="store_true",
                        help="OpenSim Inverse Kinematics")
    parser.add_argument("--s2d-augment",  action="store_true",
                        help="LSTM marker augmentation for IK")
    parser.add_argument("--s2d-side",     default="auto front",
                        help="Camera-visible side: 'auto front', 'left', 'right'")
    parser.add_argument("--s2d-realtime", action="store_true",
                        help="Show Sports2D real-time window + graphs on screen")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # Directory layout
    output_dir   = "Output"
    sports2d_dir = os.path.join(output_dir, "Sports2D")
    os.makedirs(output_dir, exist_ok=True)

    def op(name):
        return os.path.join(output_dir, name)

    json_out         = op("data_output.json")
    csv_out          = op("bio_metrics.csv")
    trc_out          = op("player_markers.trc")
    mot_out          = op("joint_angles.mot")
    report_out       = op("report.txt")
    custom_video_out = op("custom_annotated.mp4")

    # When Sports2D is used, the player pick happens inside Sports2D (on_click).
    # We skip our own picker at init time and let run_sports2d() re-seed the
    # custom tracker from the TRC file afterward.
    run_our_picker = args.pick and not args.sports2d

    print("\n" + "=" * 60)
    print("  INITIALISING ANALYZER")
    print("=" * 60)

    analyzer = SportsAnalyzer(
        video_path        = args.video,
        output_video_path = custom_video_out,
        player_id         = args.player,
        fps_override      = args.fps,
        pick              = run_our_picker,
        yolo_size         = args.yolo_size,
        player_height_m   = args.height,
    )

    # STEP 1 — Sports2D
    if args.sports2d:
        print("\n" + "=" * 60)
        print("  STEP 1 — Sports2D Pipeline")
        print("  Graphs, annotated video, MOT/TRC files")
        print("=" * 60)

        if not HAS_SPORTS2D:
            print("\n[WARN] Sports2D not installed — skipping.")
            print("       Run: pip install sports2d pose2sim\n")
        else:
            person_ordering = "on_click" if args.pick else "greatest_displacement"
            analyzer.run_sports2d(
                result_dir          = sports2d_dir,
                mode                = args.s2d_mode,
                show_realtime       = args.s2d_realtime,
                person_ordering     = person_ordering,
                do_ik               = args.s2d_ik,
                use_augmentation    = args.s2d_augment,
                visible_side        = args.s2d_side,
                participant_mass_kg = args.mass,
            )
            # run_sports2d() re-seeds the custom tracker from TRC automatically.

    # STEP 2 — Custom tracking + biomechanics
    print("\n" + "=" * 60)
    print("  STEP 2 — Custom Tracking + Biomechanics")
    print("  Skeleton overlay video, report, JSON, CSV")
    print("=" * 60)

    analyzer.process_video()

    # STEP 3 — Data export
    print("\n" + "=" * 60)
    print("  STEP 3 — Data Export")
    print("=" * 60)

    analyzer.export_unified(
        json_path = json_out,
        csv_path  = csv_out,
        trc_path  = trc_out,
        mot_path  = mot_out,
    )

    # STEP 4 — Report
    report_str = analyzer.get_report_string()
    print("\n" + report_str)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[EXPORT] Report -> {report_out}")

    # Final summary
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)

    if args.sports2d and analyzer.sports2d_runner:
        s2d  = analyzer.sports2d_runner.outputs
        vids = s2d.get("annotated_video_final", s2d.get("annotated_video", []))
        print("\n  Sports2D outputs  (Output/Sports2D/)")
        for v in vids:
            print(f"    {v}   <- annotated video")
        for f in s2d.get("angle_plots_png", []):
            print(f"    {f}   <- angle graph (PNG)")
        for f in s2d.get("mot_angles", []):
            print(f"    {f}   <- joint angles (MOT)")
        for f in s2d.get("trc_pose_m", []):
            print(f"    {f}   <- keypoints in metres (TRC)")
        if args.s2d_ik:
            for f in s2d.get("osim_model", []):
                print(f"    {f}   <- OpenSim model")
            for f in s2d.get("osim_mot", []):
                print(f"    {f}   <- OpenSim IK motion")

    print(f"\n  Custom outputs  (Output/)")
    print(f"    {custom_video_out}   <- skeleton overlay video")
    print(f"    {json_out}")
    print(f"    {csv_out}")
    print(f"    {trc_out}")
    print(f"    {mot_out}")
    print(f"    {report_out}")
    print()


if __name__ == "__main__":
    main()
