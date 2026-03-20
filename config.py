"""
Configuration for PiCrawler agent.
Change AGENT_ID and IPs per robot.
Create .env from .env.example: cp .env.example .env
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import socket

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# Agent identity (change per Pi)
AGENT_ID = os.getenv("AGENT_ID", "crawler-alpha-patrol-001")

# Local IP (auto-detect for ADS registration)
def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = os.getenv("LOCAL_IP", _get_local_ip())

# Local ports
AGENT_PORT = int(os.getenv("AGENT_PORT", "9004"))
STREAM_PORT = int(os.getenv("STREAM_PORT", "9000"))
STREAM_BASE_URL = os.getenv("STREAM_BASE_URL", f"http://localhost:{STREAM_PORT}")

# LLM (format: "provider/model", e.g. "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Mock mode: True = only log, no hardware movement
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")

# ADS registration (empty = skip registration)
ADS_ADDRESS = os.getenv("ADS_ADDRESS", "")          # e.g. "10.229.117.154:8888"
OASF_ADDRESS = os.getenv("OASF_ADDRESS", "")        # e.g. "10.229.117.154:31234"

# Speaker: TTS and sound effects (default OFF)
SPEAKER_ENABLED = os.getenv("SPEAKER_ENABLED", "false").lower() in ("true", "1", "yes")

# Calibration (measure and adjust per robot)
STEP_DISTANCE_CM = float(os.getenv("STEP_DISTANCE_CM", "2.5"))
DEGREES_PER_TURN_STEP = float(os.getenv("DEGREES_PER_TURN_STEP", "18.0"))
OBSTACLE_THRESHOLD_CM = float(os.getenv("OBSTACLE_THRESHOLD_CM", "5.0"))
PATROL_SPEED = int(os.getenv("PATROL_SPEED", "50"))
