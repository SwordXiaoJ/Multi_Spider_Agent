"""
Camera and detection client for PiCrawler.

Calls remote_stream.py's HTTP API on localhost:9000.
No direct Vilib import needed — all via HTTP.
"""

import logging
from typing import Optional

import requests
import numpy as np
import cv2

from agent_picrawler.config import STREAM_BASE_URL

logger = logging.getLogger(__name__)


class CrawlerCamera:
    """
    Interfaces with remote_stream.py's Flask endpoints.

    remote_stream.py must be running on localhost:9000.
    Camera is always real (HTTP only, no hardware risk).
    """

    def __init__(self, base_url: str = STREAM_BASE_URL, timeout: float = 5.0):
        self.base_url = base_url
        self.timeout = timeout

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Grab a single frame from the camera.
        GET /photo returns raw JPEG bytes.
        """
        try:
            resp = requests.get(f"{self.base_url}/photo", timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"grab_frame failed: HTTP {resp.status_code}")
                return None

            buf = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                logger.debug(f"Frame grabbed: {frame.shape[1]}x{frame.shape[0]}")
            return frame

        except requests.RequestException as e:
            logger.warning(f"grab_frame error: {e}")
            return None

    def get_detection_status(self) -> dict:
        """
        Get current detection status from Vilib.
        GET /status returns JSON with face/color/qr detection info.
        """
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"get_detection_status error: {e}")
            return {}

    def enable_face_detect(self, on: bool = True):
        """Toggle face detection."""
        action = "on" if on else "off"
        try:
            requests.post(f"{self.base_url}/cmd/face/{action}", timeout=self.timeout)
            logger.info(f"Face detection: {action}")
        except requests.RequestException as e:
            logger.warning(f"enable_face_detect error: {e}")

    def enable_color_detect(self, color: str):
        """
        Set color detection.
        color: 'red','orange','yellow','green','blue','purple','close'
        """
        try:
            requests.post(f"{self.base_url}/cmd/color/{color}", timeout=self.timeout)
            logger.info(f"Color detection: {color}")
        except requests.RequestException as e:
            logger.warning(f"enable_color_detect error: {e}")

    def enable_qr_detect(self, on: bool = True):
        """Toggle QR code detection."""
        action = "on" if on else "off"
        try:
            requests.post(f"{self.base_url}/cmd/qr/{action}", timeout=self.timeout)
            logger.info(f"QR detection: {action}")
        except requests.RequestException as e:
            logger.warning(f"enable_qr_detect error: {e}")

    def take_photo(self) -> Optional[str]:
        """Take a photo and save to Pi's Pictures folder."""
        try:
            resp = requests.post(f"{self.base_url}/cmd/photo", timeout=self.timeout)
            data = resp.json()
            path = data.get("photo", "")
            logger.info(f"Photo saved: {path}")
            return path
        except requests.RequestException as e:
            logger.warning(f"take_photo error: {e}")
            return None
