"""
PiCrawler Agent — entry point.

Starts A2A server with HTTP endpoints.
Routes requests to LangGraph-based mission processor.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from uuid import uuid4

from uvicorn import Config, Server
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    UnsupportedOperationError, JSONRPCResponse,
    ContentTypeNotSupportedError, InternalError,
    Message, Role, Part, TextPart,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError

from agent_picrawler.config import AGENT_ID, AGENT_PORT, MOCK_MODE, STREAM_BASE_URL, ADS_ADDRESS, OASF_ADDRESS
from agent_picrawler.capabilities import get_capabilities_json
from agent_picrawler.hardware import CrawlerControl
from agent_picrawler.brain import LLMClient
from agent_picrawler.graph import MissionManager
from agent_picrawler.card import AGENT_CARD, register_to_ads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

_start_time = time.time()
_manager: MissionManager = None


# ── A2A Executor (router) ──────────────────────────────────

class CrawlerAgentExecutor(AgentExecutor):
    """Routes A2A requests to Agent mode or Execute mode."""

    def __init__(self, manager: MissionManager):
        self.manager = manager
        self.agent_card = AGENT_CARD.model_dump(mode="json", exclude_none=True)
        self.busy = False
        self.current_task = None
        self._current_future: asyncio.Future | None = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context or not context.message or not context.message.parts:
            await event_queue.enqueue_event(
                JSONRPCResponse(error=ContentTypeNotSupportedError())
            )
            return

        prompt = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        try:
            metadata = context.message.metadata or {}

            # Handle interrupt
            if metadata.get("interrupt_current") and self.busy:
                logger.info("Interrupt requested")
                self.manager.stop_patrol()
                if self._current_future and not self._current_future.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(self._current_future), timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                self.busy = False
                self.current_task = None

            is_agent = metadata.get("central_detection", False)

            if is_agent:
                result = await self._agent_mode(context.message, metadata)
            else:
                result = await self._execute_mode(context.message, metadata)

            # Format and send response
            output = self._format_output(result)
            msg = Message(
                message_id=str(uuid4()),
                role=Role.agent,
                metadata={"name": self.agent_card["name"]},
                parts=[Part(TextPart(text=output))],
            )
            await event_queue.enqueue_event(msg)

        except Exception as e:
            logger.error(f"A2A error: {e}")
            raise ServerError(error=InternalError()) from e

    async def _agent_mode(self, message: Message, metadata: dict) -> dict:
        """Agent mode: register mission, LLM decides everything."""
        mission_id = metadata.get("mission_id", "")
        task_goal = metadata.get("task_goal", "")
        callback_url = metadata.get("callback_url", "")
        steps = metadata.get("steps", [])

        if mission_id and task_goal:
            self.manager.set_mission(mission_id, task_goal, callback_url)
            logger.info(f"Agent mode: mission={mission_id}, goal={task_goal}")

        await self.manager.decide_initial_actions()

        # If search step, start patrol
        search_step = next((s for s in steps if s.get("task_type") == "search"), None)
        if search_step:
            self.busy = True
            self.current_task = "patrol"
            try:
                pattern = search_step.get("search_pattern", "lawnmower")
                self._current_future = asyncio.ensure_future(self.manager.patrol(pattern))
                return await self._current_future
            finally:
                self.busy = False
                self.current_task = None
                self._current_future = None

        logger.info("Agent mode: stationary, waiting for observations")
        return {
            "request_id": message.message_id,
            "agent_id": AGENT_ID,
            "detections": [],
            "area_covered_pct": 0.0,
            "mission_time_ms": 0,
            "search_pattern": "none",
            "waypoints_completed": 0,
            "waypoints_total": 0,
            "summary": "Mission registered. Waiting for observations.",
            "mission_status": "awaiting_observations",
        }

    async def _execute_mode(self, message: Message, metadata: dict) -> dict:
        """Execute mode: run steps sequentially."""
        steps = metadata.get("steps", [])
        if not steps:
            action = metadata.get("action", "stop")
            steps = [{"action": action, "description": f"Direct: {action}"}]

        logger.info(f"Execute mode: {len(steps)} steps")
        self.busy = True
        self.current_task = steps[0].get("description", "execute")
        try:
            result = await self.manager.execute_steps(steps)
            result["request_id"] = message.message_id
            return result
        finally:
            self.busy = False
            self.current_task = None

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    def _format_output(self, result: dict) -> str:
        lines = [
            f"Agent: {result.get('agent_id', AGENT_ID)}",
            f"Detections: {len(result.get('detections', []))}",
            f"Coverage: {result.get('area_covered_pct', 0):.1f}%",
            f"Time: {result.get('mission_time_ms', 0)}ms",
            f"Status: {result.get('mission_status', 'unknown')}",
        ]
        summary = result.get("summary", "")
        if summary:
            lines.append(f"Summary: {summary}")
        return "\n".join(lines)


# ── HTTP Endpoints ──────────────────────────────────────────

async def health(request: Request):
    return JSONResponse({
        "status": "ok",
        "agent_id": AGENT_ID,
        "uptime_s": int(time.time() - _start_time),
        "mock_mode": MOCK_MODE,
    })


async def agent_status(request: Request):
    data = {
        "agent_id": AGENT_ID,
        "uptime_s": int(time.time() - _start_time),
        "mock_mode": MOCK_MODE,
    }
    if _manager:
        data["standing"] = _manager.control._standing
        data["mission_active"] = _manager.mission_active
    return JSONResponse(data)


async def capabilities_endpoint(request: Request):
    data = get_capabilities_json()
    data["agent_id"] = AGENT_ID
    return JSONResponse(data)


async def observations(request: Request):
    if not _manager:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    result = await _manager.handle_observation(data)
    return JSONResponse(result)


async def camera_frame(request: Request):
    try:
        import requests as req
        resp = req.get(f"{STREAM_BASE_URL}/photo", timeout=5)
        if resp.status_code != 200:
            return JSONResponse({"error": "Camera unavailable"}, status_code=502)
        return Response(content=resp.content, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def voice_record(request: Request):
    params = request.query_params
    duration = min(int(params.get("duration", "5")), 30)
    wav_path = "/tmp/voice_command.wav"
    try:
        result = subprocess.run(
            ["arecord", "-D", "plughw:3,0", "-f", "S16_LE",
             "-r", "16000", "-c", "1", "-d", str(duration), wav_path],
            capture_output=True, timeout=duration + 5,
        )
        if result.returncode != 0:
            return JSONResponse({"error": "Recording failed"}, status_code=500)
        with open(wav_path, "rb") as f:
            audio_data = f.read()
        return Response(content=audio_data, media_type="audio/wav",
                       headers={"Content-Disposition": "attachment; filename=voice_command.wav"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Server startup ──────────────────────────────────────────

async def main():
    print("=" * 60)
    print(f"PiCrawler Agent (LangGraph)")
    print(f"Agent: {AGENT_ID}")
    print(f"Port: {AGENT_PORT}")
    print(f"Mock: {os.getenv('MOCK_MODE', 'true')}")
    print("=" * 60)

    global _manager

    control = CrawlerControl(speed=int(os.getenv("PATROL_SPEED", "50")))
    llm = LLMClient()
    _manager = MissionManager(control, llm)

    executor = CrawlerAgentExecutor(_manager)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=AGENT_CARD,
        http_handler=request_handler,
    )

    app = server.build()
    app.add_route("/health", health, methods=["GET"])
    app.add_route("/status", agent_status, methods=["GET"])
    app.add_route("/capabilities", capabilities_endpoint, methods=["GET"])
    app.add_route("/observations", observations, methods=["POST"])
    app.add_route("/camera/frame", camera_frame, methods=["GET"])
    app.add_route("/voice/record", voice_record, methods=["POST"])

    # ADS self-registration (non-blocking, failure is non-fatal)
    if ADS_ADDRESS and OASF_ADDRESS:
        register_to_ads(ADS_ADDRESS, OASF_ADDRESS)

    config = Config(app=app, host="0.0.0.0", port=AGENT_PORT, loop="asyncio")
    await Server(config).serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
