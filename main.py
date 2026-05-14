#!/usr/bin/env python3
"""CLI entry point for EXPAI Inference Platform.

Usage:
    python main.py --module epp_gloves --video videos/test.mp4 --model models/epp_gloves_v1.pt
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from core.config import BASE_DIR, load_module_config, resolve_path
from core.inference_engine import InferenceEngine
from core.video_processor import VideoProcessor
from core.visualizer import Visualizer
from modules.epp_gloves.module import EPPGlovesModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXPAI Inference Platform CLI")
    parser.add_argument("--module", default="epp_gloves", help="Module name (default: epp_gloves)")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--config", default=None, help="Path to module config YAML (optional)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--no-video-output", action="store_true", help="Skip writing annotated video")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    video_path = resolve_path(args.video)
    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    # Load module
    if args.module != "epp_gloves":
        logger.error(f"Module '{args.module}' not yet implemented")
        return 1

    module = EPPGlovesModule(config_path=args.config)
    # Override model path in config
    module.config["model"]["path"] = str(model_path)

    engine = InferenceEngine.from_module_config(module.config)
    visualizer = Visualizer()

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    job_id = video_path.stem
    annotated_path = annotated_dir / f"{job_id}_annotated.mp4"

    logger.info(f"Starting inference | module={args.module} | video={video_path.name}")

    with VideoProcessor(video_path) as video:
        writer = None
        if not args.no_video_output:
            writer = video.get_writer(annotated_path)

        for frame_idx, timestamp, frame in video.frames():
            detections = engine.predict(frame)
            result = module.process_frame(frame, detections, frame_idx, timestamp)

            visualizer.draw_detections(frame, detections)
            visualizer.draw_status_overlay(
                frame,
                operator_state=result["annotations"]["operator_state"],
                compliance_pct=result["annotations"]["compliance_pct"],
                frame_index=frame_idx,
                timestamp=timestamp,
                alert_active=result["annotations"]["alert_active"],
            )

            if writer:
                writer.write(frame)

            if frame_idx % 100 == 0:
                pct = (frame_idx / max(video.total_frames, 1)) * 100
                logger.info(f"Progress: {pct:.1f}% | F={frame_idx} | state={result['annotations']['operator_state']}")

        if writer:
            writer.release()

    # Report
    report = module.generate_report()
    report_file = report_dir / f"{job_id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    analysis = report.get("analysis", {})
    logger.info("=" * 50)
    logger.info(f"Compliance: {analysis.get('compliance_percentage', 0):.1f}%")
    logger.info(f"Incidents:  {analysis.get('incident_count', 0)}")
    logger.info(f"Report:     {report_file}")
    if not args.no_video_output:
        logger.info(f"Video:      {annotated_path}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
