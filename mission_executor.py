"""
Mission executor: combines movement, camera, and LLM to run a patrol mission.

All synchronous — PiCrawler is a synchronous API.
"""

import time
import logging
from typing import List, Dict, Any

import requests

from agent_picrawler.config import CENTRAL_URL, CENTRAL_GATEWAY_PORT, AGENT_ID, MOCK_MODE
from agent_picrawler.crawler_control import CrawlerControl
from agent_picrawler.crawler_camera import CrawlerCamera
from agent_picrawler.llm_client import LLMClient
from agent_picrawler.search_patterns import generate_pattern

logger = logging.getLogger(__name__)


class MissionExecutor:
    """Executes a complete patrol mission."""

    def __init__(
        self,
        control: CrawlerControl,
        camera: CrawlerCamera,
        llm: LLMClient,
    ):
        self.control = control
        self.camera = camera
        self.llm = llm
        self._stop_requested = False

    def stop(self):
        """Request mission stop."""
        self._stop_requested = True

    def execute(self, mission_config: dict) -> dict:
        """
        Execute a full patrol mission.

        mission_config: {
            "pattern": "lawnmower",
            "target": "person",
            "width_cm": 200,
            "height_cm": 200,
            "spacing_cm": 30,
            "detect_face": True,
            "detect_color": "",   # "red", "blue", etc. or empty
        }

        Returns mission result dict.
        """
        self._stop_requested = False
        start_time = time.time()

        pattern = mission_config.get("pattern", "lawnmower")
        target = mission_config.get("target", "any")
        detect_face = mission_config.get("detect_face", True)
        detect_color = mission_config.get("detect_color", "")

        # Generate waypoints
        pattern_kwargs = {}
        if pattern == "lawnmower":
            pattern_kwargs = {
                "width_cm": mission_config.get("width_cm", 200),
                "height_cm": mission_config.get("height_cm", 200),
                "spacing_cm": mission_config.get("spacing_cm", 30),
            }
        elif pattern == "spiral":
            pattern_kwargs = {
                "radius_cm": mission_config.get("radius_cm", 100),
                "spacing_cm": mission_config.get("spacing_cm", 30),
            }
        elif pattern == "expanding_square":
            pattern_kwargs = {
                "size_cm": mission_config.get("size_cm", 200),
                "spacing_cm": mission_config.get("spacing_cm", 30),
            }

        waypoints = generate_pattern(pattern, **pattern_kwargs)
        total_waypoints = len(waypoints)
        logger.info(f"Mission started: pattern={pattern}, target={target}, "
                    f"waypoints={total_waypoints}")

        # Enable detections
        if detect_face:
            self.camera.enable_face_detect(True)
        if detect_color:
            self.camera.enable_color_detect(detect_color)

        detections: List[Dict[str, Any]] = []
        waypoints_completed = 0

        try:
            # Stand up
            self.control.stand()
            self.control.reset_position()

            for i, (tx, ty) in enumerate(waypoints):
                if self._stop_requested:
                    logger.info("Mission stop requested")
                    break

                logger.info(f"Waypoint {i+1}/{total_waypoints}: ({tx:.1f}, {ty:.1f})")

                # Navigate to waypoint
                reached = self.control.navigate_to(tx, ty)
                waypoints_completed = i + 1

                if not reached:
                    logger.warning(f"Could not reach waypoint {i+1}, skipping")

                # Detect at current position
                new_dets = self._detect_at_position(target)
                detections.extend(new_dets)

                # Report progress to central (non-blocking)
                self._report_progress(waypoints_completed, total_waypoints, detections)

            # Sit down
            self.control.sit()

        finally:
            # Always clean up detections, even if killed/crashed
            if detect_face:
                self.camera.enable_face_detect(False)
            if detect_color:
                self.camera.enable_color_detect("close")
            logger.info("Detection cleanup done")

        elapsed_ms = int((time.time() - start_time) * 1000)
        coverage = (waypoints_completed / max(total_waypoints, 1)) * 100

        # LLM summary
        stats = {
            "pattern": pattern,
            "waypoints_completed": waypoints_completed,
            "waypoints_total": total_waypoints,
            "coverage_pct": coverage,
            "elapsed_ms": elapsed_ms,
        }
        summary = self.llm.summarize_mission(detections, stats)

        result = {
            "agent_id": AGENT_ID,
            "detections": detections,
            "area_covered_pct": coverage,
            "mission_time_ms": elapsed_ms,
            "search_pattern": pattern,
            "waypoints_completed": waypoints_completed,
            "waypoints_total": total_waypoints,
            "summary": summary,
        }

        logger.info(f"Mission complete: {len(detections)} detections, "
                    f"{coverage:.0f}% covered, {elapsed_ms}ms")
        return result

    def _detect_at_position(self, target: str) -> List[Dict[str, Any]]:
        """Run detection at current position."""
        detections = []
        x, y, heading = self.control.get_position()

        # Grab a frame (for future YOLO, or just to have it)
        frame = self.camera.grab_frame()

        # Check Vilib detection status
        status = self.camera.get_detection_status()
        if not status:
            return detections

        # Check face detection
        face_count = status.get("face", {}).get("count", 0)
        if face_count > 0:
            det = {
                "label": "face",
                "confidence": 0.8,
                "x_cm": x,
                "y_cm": y,
                "source": "vilib_face",
                "details": status.get("face", {}),
            }
            # Ask LLM if this matches the target
            evaluation = self.llm.evaluate_detection(target, det)
            det["llm_match"] = evaluation.get("match", False)
            det["llm_reason"] = evaluation.get("reason", "")
            detections.append(det)
            logger.info(f"Face detected at ({x:.1f}, {y:.1f}), "
                        f"match={det['llm_match']}")

        # Check color detection
        color_count = status.get("color", {}).get("count", 0)
        if color_count > 0:
            det = {
                "label": "color_object",
                "confidence": 0.7,
                "x_cm": x,
                "y_cm": y,
                "source": "vilib_color",
                "details": status.get("color", {}),
            }
            evaluation = self.llm.evaluate_detection(target, det)
            det["llm_match"] = evaluation.get("match", False)
            det["llm_reason"] = evaluation.get("reason", "")
            detections.append(det)
            logger.info(f"Color object detected at ({x:.1f}, {y:.1f}), "
                        f"match={det['llm_match']}")

        # Check QR code
        qr_data = status.get("qrcode", {}).get("data", "None")
        if qr_data != "None":
            det = {
                "label": "qr_code",
                "confidence": 1.0,
                "x_cm": x,
                "y_cm": y,
                "source": "vilib_qr",
                "qr_data": qr_data,
            }
            detections.append(det)
            logger.info(f"QR code at ({x:.1f}, {y:.1f}): {qr_data}")

        return detections

    def _report_progress(self, completed: int, total: int, detections: list):
        """Report progress to central dashboard (non-blocking)."""
        try:
            x, y, heading = self.control.get_position()
            data = {
                "drone_id": AGENT_ID,
                "status": "searching",
                "latitude": y,   # use y_cm as lat proxy
                "longitude": x,  # use x_cm as lon proxy
            }
            requests.post(
                f"{CENTRAL_URL}:{CENTRAL_GATEWAY_PORT}/api/drones/update",
                json=data,
                timeout=2,
            )
        except Exception:
            pass  # don't fail mission if central is unreachable
