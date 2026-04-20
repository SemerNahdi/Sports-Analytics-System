#!/usr/bin/env python3
"""
Juventus Sports Analytics System — Entry Point v5
====================================================

A hybrid 2D sports biomechanics pipeline combining custom tracking 
with Sports2D for high-quality joint angles, gait analysis, and injury risk metrics.

USAGE:
  python run_analysis.py --video match.mp4 --sports2d --s2d-pick
  python run_analysis.py --video match.mp4 --sports2d --s2d-pick --s2d-ik --s2d-augment --height 1.82 --mass 78

RECOMMENDED (maximum value):
  --sports2d --s2d-pick          → always use Sports2D picker + unified outputs
"""

import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from pathlib import Path

from src.analytics.sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Juventus Sports Analytics v5 — Hybrid Biomechanics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Recommended: --sports2d --s2d-pick"
    )

    # Core
    core = parser.add_argument_group("Core Options")
    core.add_argument("--video", required=True, help="Path to input video")
    core.add_argument("--output", default="output_annotated.mp4",
                      help="Output annotated video filename (default: output_annotated.mp4)")
    core.add_argument("--player", type=int, default=1, help="Player ID label")
    core.add_argument("--fps", type=float, help="Override detected FPS")
    core.add_argument("--stride", type=int, default=int(os.getenv("ANALYSIS_STRIDE", "2")),
                      help="Process every Nth frame (default: 2 or $ANALYSIS_STRIDE)")
    core.add_argument("--target-height", type=int, default=int(os.getenv("ANALYSIS_TARGET_HEIGHT", "640")),
                      help="Downscale video to this height before analysis (default: 640 or $ANALYSIS_TARGET_HEIGHT)")
    core.add_argument("--height", type=float, default=1.75,
                      help="Player height in metres (default: 1.75)")
    core.add_argument("--mass", type=float, default=75.0,
                      help="Player mass in kg (default: 75.0)")

    # Tracker
    tracker = parser.add_argument_group("Tracking Options")
    tracker.add_argument("--pick", action="store_true",
                         help="Use custom interactive picker (only if not using --sports2d)")
    tracker.add_argument("--yolo-size", default=os.getenv("YOLO_SIZE_DEFAULT", "n"), choices=["n", "s", "m", "l", "x"],
                         help="YOLO model size for custom tracker (default: n or $YOLO_SIZE_DEFAULT)")

    # Sports2D
    s2d = parser.add_argument_group("Sports2D Options (Recommended)")
    s2d.add_argument("--sports2d", action="store_true",
                     help="Run Sports2D pipeline (recommended for best graphs and IK)")
    s2d.add_argument("--s2d-pick", action="store_true",
                     help="Use Sports2D interactive on_click player picker")
    s2d.add_argument("--s2d-mode", default="balanced",
                     choices=["lightweight", "balanced", "performance"],
                     help="Sports2D pose mode (default: balanced)")
    s2d.add_argument("--s2d-ik", action="store_true",
                     help="Enable OpenSim Inverse Kinematics")
    s2d.add_argument("--s2d-augment", action="store_true",
                     help="Use LSTM marker augmentation for IK")
    s2d.add_argument("--s2d-side", default="auto front",
                     help="Visible side: 'auto front', 'left', 'right'")
    s2d.add_argument("--s2d-realtime", action="store_true",
                     help="Show Sports2D real-time window")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"\n[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    # ── Directory Setup ─────────────────────────────────────────────────────
    output_dir = Path("Output")
    sports2d_dir = output_dir / "Sports2D"
    results_dir = output_dir / "results"

    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    sports2d_dir.mkdir(exist_ok=True)

    # Output paths
    video_out = output_dir / args.output
    json_out = output_dir / "data_output.json"
    csv_out = output_dir / "bio_metrics.csv"
    trc_out = output_dir / "player_markers.trc"
    mot_out = output_dir / "joint_angles.mot"
    report_out = output_dir / "report.txt"

    print("\n" + "=" * 70)
    print("   JUVENTUS SPORTS ANALYTICS v5")
    print("=" * 70)
    print(f"Video : {Path(args.video).name}")
    print(f"Mode  : {'Sports2D + Custom' if args.sports2d else 'Custom Only'}")
    if args.sports2d and args.s2d_pick:
        print("Picker: Sports2D on_click (recommended)")
    print("-" * 70)

    # ── Step 1: Custom Tracking + Biomechanics ───────────────────────────────
    print("\n[STEP 1/5] Initialising Custom Tracker + Biomechanics Engine")
    analyzer = SportsAnalyzer(
        video_path=args.video,
        output_video_path=str(video_out),
        player_id=args.player,
        fps_override=args.fps,
        pick=args.pick,
        yolo_size=args.yolo_size,
        player_height_m=args.height,
    )

    analyzer.process_video(stride=args.stride, target_height=args.target_height)

    # ── Step 2: Sports2D Pipeline (if requested) ─────────────────────────────
    if args.sports2d:
        print("\n[STEP 2/5] Running Sports2D Pipeline")
        if not HAS_SPORTS2D:
            print("   ⚠  Sports2D not installed — skipping.")
            print("      pip install sports2d pose2sim")
        else:
            person_ordering = "on_click" if args.s2d_pick else "greatest_displacement"
            analyzer.run_sports2d(
                result_dir=str(sports2d_dir),
                mode=args.s2d_mode,
                show_realtime=args.s2d_realtime,
                person_ordering=person_ordering,
                do_ik=args.s2d_ik,
                use_augmentation=args.s2d_augment,
                visible_side=args.s2d_side,
                participant_mass_kg=args.mass,
            )

    # ── Step 3: Unified Export ───────────────────────────────────────────────
    print("\n[STEP 3/5] Exporting Unified Data (JSON + CSV + OpenSim)")
    analyzer.export_unified(
        json_path=str(json_out),
        csv_path=str(csv_out),
        trc_path=str(trc_out),
        mot_path=str(mot_out),
    )

    # ── Step 4: Generate Plots ───────────────────────────────────────────────
    print("\n[STEP 4/5] Generating Analytical Plots → Output/results/")
    plotter = AnalyticsPlotter(results_dir=str(results_dir), player_id=args.player)
    plotter.generate_all(
        frame_metrics=analyzer.frame_metrics,
        bio_engine=analyzer.bio_engine,
    )

    # ── Step 5: Generate Report ──────────────────────────────────────────────
    print("\n[STEP 5/5] Generating Final Report")
    report_str = analyzer.get_report_string()
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_str)

    # ── Final Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("   ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\n📁 Main Outputs:")
    print(f"   • Annotated Video     : {video_out.name}")
    print(f"   • Unified JSON        : {json_out.name}")
    print(f"   • Unified CSV         : {csv_out.name}")
    print(f"   • OpenSim TRC         : {trc_out.name}")
    print(f"   • OpenSim MOT         : {mot_out.name}")
    print(f"   • Report              : {report_out.name}")

    print(f"\n📊 Plots saved to: Output/results/")

    if args.sports2d and getattr(analyzer, 'sports2d_runner', None) and analyzer.sports2d_runner.outputs:
        print(f"\n🏟️  Sports2D Outputs (Output/Sports2D/):")
        s2d = analyzer.sports2d_runner.outputs
        for key, files in s2d.items():
            if files:
                if isinstance(files, list):
                    for f in files[:3]:  # limit display
                        print(f"   • {Path(f).name}")
                else:
                    print(f"   • {Path(str(files)).name}")

    print(f"\n✅ Done. All files saved in './Output/' folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()