#!/usr/bin/env python3
"""
Juventus Sports Analytics System — Entry Point
================================================
Usage:
    # Auto-select the most persistent player:
    python run_analysis.py --video match.mp4

        # Click to choose which player to track:
    python run_analysis.py --video match.mp4 --pick

    # Full options:
    python run_analysis.py --video clip.mp4 --pick --player 10 --output out.mp4
"""

import argparse
import os
import sys
import io
import contextlib
from sports_analytics import SportsAnalyzer


def main():
    output_dir = "Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = argparse.ArgumentParser(
        description="Juventus Sports Analytics — single-player biomechanical analysis")
    parser.add_argument("--video",  type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default=os.path.join(output_dir, "output_annotated.mp4"),
                        help=f"Path for annotated output video (default: {output_dir}/output_annotated.mp4)")
    parser.add_argument("--player", type=int, default=1,
                        help="Player jersey/ID label shown on screen (default: 1)")
    parser.add_argument("--fps",    type=float, default=None,
                        help="Override video FPS (optional)")
    parser.add_argument("--json",   type=str, default=os.path.join(output_dir, "metrics.json"),
                        help=f"JSON export path (default: {output_dir}/metrics.json)")
    parser.add_argument("--csv",    type=str, default=os.path.join(output_dir, "metrics.csv"),
                        help=f"CSV export path (default: {output_dir}/metrics.csv)")
    parser.add_argument("--pick",   action="store_true",
                        help="Open interactive window to click and choose which player to track "
                             "(default: auto-selects the most persistent player)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    analyzer = SportsAnalyzer(
        video_path=args.video,
        output_video_path=args.output,
        player_id=args.player,
        fps_override=args.fps,
        pick=args.pick,
    )

    summary = analyzer.process_video()

    # Capture terminal summary
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        analyzer.print_report()
        df = analyzer.get_dataframe()
        print(f"[INFO] DataFrame: {df.shape[0]} frames × {df.shape[1]} metrics")
        print(df[["frame_idx","timestamp","speed","cadence","stride_length",
                  "left_knee_angle","right_knee_angle","l_valgus","r_valgus",
                  "risk_score","gait_symmetry"]].head(8).to_string(index=False))

    captured_summary = f.getvalue()
    
    # Print the summary to terminal as well
    print(captured_summary)
    
    # Save captured summary to file
    summary_path = os.path.join(output_dir, "terminal_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(captured_summary)
    print(f"[INFO] Terminal summary saved to {summary_path}")

    # Export data files
    analyzer.export_json(args.json)
    analyzer.export_csv(args.csv)

    print(f"\n[DONE] Annotated video → {args.output}")


if __name__ == "__main__":
    main()
