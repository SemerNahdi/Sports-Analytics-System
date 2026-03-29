#!/usr/bin/env python3
"""
Juventus Sports Analytics System — Entry Point  v5
====================================================
INSTALL:
  pip install ultralytics opencv-python numpy pandas scipy matplotlib
  pip install sports2d pose2sim          # for Sports2D pipeline
  # ffmpeg must be on PATH for H.264 re-encoding (https://ffmpeg.org)
  # For OpenSim IK: conda install -c opensim-org opensim

USAGE — basic:
  python run_analysis.py --video match.mp4

USAGE — with interactive player picker:
  python run_analysis.py --video match.mp4 --pick

USAGE — with Sports2D pipeline:
  python run_analysis.py --video match.mp4 --sports2d

USAGE — Sports2D with OpenSim IK:
  python run_analysis.py --video match.mp4 --sports2d --s2d-ik --s2d-augment

USAGE — everything:
  python run_analysis.py --video match.mp4 --pick --sports2d \\
      --s2d-ik --s2d-augment --height 1.82 --mass 78

OUTPUTS  (Output/ folder):
  output_annotated.mp4    annotated video — skeleton overlay on clean video
  data_output.json        unified hierarchical data (all sources merged)
  bio_metrics.csv         unified flat time-series (all sources merged)
  player_markers.trc      OpenSim marker trajectory file
  joint_angles.mot        OpenSim joint angle motion file
  report.txt              full text report

  Output/results/         high-resolution plots (PNG + SVG)
    speed_acceleration_profile.png/.svg
    joint_angles_timeseries.png/.svg
    risk_scores.png/.svg
    metabolic_power.png/.svg
    knee_flexion.png/.svg
    clinical_valgus.png/.svg
    hip_ankle_kinematics.png/.svg
    angular_velocities.png/.svg
    gait_events.png/.svg
    arm_swing.png/.svg

  Output/Sports2D/        (--sports2d only)
    *_h264.mp4            annotated video (H.264)
    *.mot                 Sports2D joint angle timeseries
    *_m.trc               Sports2D keypoints in metres
    *.png                 Sports2D angle plots
    *.toml                camera calibration
    *.c3d                 C3D biomechanics file
    *.osim / *_ik.mot     OpenSim model + IK motion (--s2d-ik only)
"""

import argparse
import os
import sys

from sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D


def main():
    parser = argparse.ArgumentParser(
        description="Juventus Sports Analytics v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    parser.add_argument("--video",       required=True,  help="Input video path")
    parser.add_argument("--output",      default="output_annotated.mp4",
                        help="Annotated video output filename")
    parser.add_argument("--player",      type=int, default=1, help="Player ID label")
    parser.add_argument("--fps",         type=float, default=None,
                        help="Override detected FPS")
    parser.add_argument("--height",      type=float, default=1.75,
                        help="Player height in metres (default 1.75)")
    parser.add_argument("--mass",        type=float, default=75.0,
                        help="Player mass in kg (default 75.0)")

    # ── Tracker ───────────────────────────────────────────────────────────────
    parser.add_argument("--pick",        action="store_true",
                        help="Interactively click to select the player to track")
    parser.add_argument("--yolo-size",   default="m", choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size (default: m)")

    # ── Sports2D pipeline ─────────────────────────────────────────────────────
    parser.add_argument("--sports2d",    action="store_true",
                        help="Run Sports2D.process() after our pipeline")
    parser.add_argument("--s2d-mode",    default="balanced",
                        choices=["lightweight", "balanced", "performance"],
                        help="Sports2D pose mode (default: balanced)")
    parser.add_argument("--s2d-pick",    action="store_true",
                        help="Sports2D interactive player picker (on_click vs auto)")
    parser.add_argument("--s2d-ik",      action="store_true",
                        help="Run OpenSim Inverse Kinematics (requires OpenSim install)")
    parser.add_argument("--s2d-augment", action="store_true",
                        help="LSTM marker augmentation for improved IK quality")
    parser.add_argument("--s2d-side",    default="auto front",
                        help="Visible side for IK: 'auto front', 'left', 'right' (default: 'auto front')")
    parser.add_argument("--s2d-realtime", action="store_true",
                        help="Show Sports2D real-time processing window")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # ── Directory layout ──────────────────────────────────────────────────────
    output_dir   = "Output"
    sports2d_dir = os.path.join(output_dir, "Sports2D")
    results_dir  = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    def op(name):
        return os.path.join(output_dir, os.path.basename(name))

    video_out  = op(args.output)
    json_out   = op("data_output.json")
    csv_out    = op("bio_metrics.csv")
    trc_out    = op("player_markers.trc")
    mot_out    = op("joint_angles.mot")
    report_out = op("report.txt")

    # ── Step 1: Tracking + biomechanics ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — Tracking + Biomechanics Engine")
    print("=" * 60)

    analyzer = SportsAnalyzer(
        video_path        = args.video,
        output_video_path = video_out,
        player_id         = args.player,
        fps_override      = args.fps,
        pick              = args.pick,
        yolo_size         = args.yolo_size,
        player_height_m   = args.height,
    )
    summary = analyzer.process_video()

    # ── Step 2: Sports2D pipeline (optional) ──────────────────────────────────
    if args.sports2d:
        print("\n" + "=" * 60)
        print("  STEP 2 — Sports2D Native Pipeline")
        print("=" * 60)

        if not HAS_SPORTS2D:
            print("\n[WARN] Sports2D is not installed — skipping.")
            print("       Run: pip install sports2d pose2sim\n")
        else:
            person_ordering = "on_click" if args.s2d_pick else "greatest_displacement"
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

    # ── Step 3: Unified data export ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — Unified Data Export")
    print("=" * 60)

    analyzer.export_unified(
        json_path = json_out,
        csv_path  = csv_out,
        trc_path  = trc_out,
        mot_path  = mot_out,
    )

    # ── Step 4: Analytics plots ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — Analytics Plots → Output/results/")
    print("=" * 60)

    plotter = AnalyticsPlotter(results_dir=results_dir, player_id=args.player)
    plotter.generate_all(
        frame_metrics = analyzer.frame_metrics,
        bio_engine    = analyzer.bio_engine,
    )

    # ── Step 5: Report ────────────────────────────────────────────────────────
    report_str = analyzer.get_report_string()
    print("\n" + report_str)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[EXPORT] Report → {report_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE — Output Files")
    print("=" * 60)
    print(f"\n  Output/")
    print(f"    {video_out}        ← annotated video (clean, no side panel)")
    print(f"    {json_out}     ← unified JSON (all data sources)")
    print(f"    {csv_out}      ← unified CSV (all data sources)")
    print(f"    {trc_out}   ← OpenSim marker trajectory")
    print(f"    {mot_out}    ← OpenSim joint angle motion")
    print(f"    {report_out}")
    print(f"\n  Output/results/    ← analytical plots (PNG + SVG)")

    if args.sports2d and analyzer.sports2d_runner and analyzer.sports2d_runner.outputs:
        s2d = analyzer.sports2d_runner.outputs
        print(f"\n  Output/Sports2D/")
        for v in s2d.get("annotated_video_h264", []):
            print(f"    {v}  ← Sports2D annotated (H.264)")
        for f in s2d.get("mot_angles", []):
            print(f"    {f}  (Sports2D angle timeseries)")
        for f in s2d.get("trc_pose_m", []):
            print(f"    {f}  (Sports2D keypoints in metres)")
        for f in s2d.get("calib_toml", []):
            print(f"    {f}  (camera calibration)")
        for f in s2d.get("c3d", []):
            print(f"    {f}  (C3D biomechanics)")
        if args.s2d_ik:
            for f in s2d.get("osim_model", []):
                print(f"    {f}  (OpenSim model)")
            for f in s2d.get("osim_mot", []):
                print(f"    {f}  (OpenSim IK motion)")
            print(f"\n  To open in OpenSim:")
            print(f"    File → Open Model → *.osim")
            print(f"    File → Load Motion → *_ik.mot")
        else:
            print(f"\n  For OpenSim IK, rerun with --s2d-ik --s2d-augment")
            print(f"  (requires: conda install -c opensim-org opensim)")

    print()


if __name__ == "__main__":
    main()
