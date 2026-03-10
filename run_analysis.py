#!/usr/bin/env python3
"""
Juventus Sports Analytics System — Entry Point
"""

import argparse
import os
import sys
from sports_analytics import SportsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Juventus Sports Analytics — single-player biomechanical analysis")
    
    parser.add_argument("--video",  type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_annotated.mp4",
                        help="Path for annotated output video")
    parser.add_argument("--player", type=int, default=1,
                        help="Player jersey/ID label")
    parser.add_argument("--fps",    type=float, default=None,
                        help="Override video FPS")
    parser.add_argument("--json",   type=str, default="metrics.json",
                        help="JSON export path")
    parser.add_argument("--csv",    type=str, default="metrics.csv",
                        help="CSV export path")
    parser.add_argument("--pick",   action="store_true",
                        help="Interactive player selection")
    # --- ADD THIS LINE BELOW ---
    parser.add_argument("--yolo-size", type=str, default="m",
                        help="YOLO model size (n, s, m, l, x)")
    
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = "Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Resolve output paths to Output/ folder
    video_out = os.path.join(output_dir, os.path.basename(args.output))
    json_out = os.path.join(output_dir, os.path.basename(args.json))
    csv_out = os.path.join(output_dir, os.path.basename(args.csv))
    report_out = os.path.join(output_dir, "report.txt")

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    analyzer = SportsAnalyzer(
        video_path=args.video,
        output_video_path=video_out,
        player_id=args.player,
        fps_override=args.fps,
        pick=args.pick,
        yolo_size=args.yolo_size
    )

    summary = analyzer.process_video()
    
    # Get report and print it
    report_str = analyzer.get_report_string()
    print("\n" + report_str + "\n")
    
    # Save report to file
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[EXPORT] Report → {report_out}")

    analyzer.export_json(json_out)
    analyzer.export_csv(csv_out)

    df = analyzer.get_dataframe()
    print(f"[INFO] DataFrame: {df.shape[0]} frames × {df.shape[1]} metrics")
    print(df[["frame_idx","timestamp","speed","cadence","stride_length",
              "left_knee_angle","right_knee_angle","l_valgus","r_valgus",
              "risk_score","gait_symmetry"]].head(8).to_string(index=False))

    print(f"\n[DONE] Results managed in: {output_dir}")
    print(f"[INFO] Annotated video → {video_out}")


if __name__ == "__main__":
    main()