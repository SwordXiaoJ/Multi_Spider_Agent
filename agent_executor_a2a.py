"""A2A Agent Executor for PiCrawler patrol agent."""

import json
import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    UnsupportedOperationError, JSONRPCResponse,
    ContentTypeNotSupportedError, InternalError,
    Message, Role, Part, TextPart,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError

from agent_picrawler.agent import CrawlerSearchAgent
from agent_picrawler.card import AGENT_CARD, AGENT_ID

logger = logging.getLogger("crawler.agent_executor")


class CrawlerAgentExecutor(AgentExecutor):
    """A2A Agent Executor for PiCrawler."""

    def __init__(self):
        self.agent = CrawlerSearchAgent(AGENT_ID)
        self.agent_card = AGENT_CARD.model_dump(mode="json", exclude_none=True)
        logger.info(f"Initialized CrawlerAgentExecutor: {AGENT_ID}")

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
            request = self._parse_request(context.message, prompt)
            result = await self.agent.search(request)
            output = self._format_output(result)

            message = Message(
                message_id=str(uuid4()),
                role=Role.agent,
                metadata={"name": self.agent_card["name"]},
                parts=[Part(TextPart(text=output))],
            )
            await event_queue.enqueue_event(message)

        except Exception as e:
            logger.error(f"Error processing search: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    def _parse_request(self, message: Message, prompt: str) -> dict:
        request = {
            "request_id": message.message_id,
            "prompt": prompt,
            "target_type": "any",
        }

        if message.metadata:
            if "target_type" in message.metadata:
                request["target_type"] = message.metadata["target_type"]
            if "search_pattern" in message.metadata:
                request["search_pattern"] = message.metadata["search_pattern"]

        # Parse structured payload from planner
        try:
            if "\nTask: " in prompt:
                _, payload_str = prompt.split("\nTask: ", 1)
                payload = json.loads(payload_str)
                if "target_type" in payload:
                    request["target_type"] = payload["target_type"]
                if "prompt" in payload:
                    request["prompt"] = payload["prompt"]
        except (json.JSONDecodeError, ValueError):
            pass

        return request

    def _format_output(self, result: dict) -> str:
        output = f"Patrol Result:\n"
        output += f"Agent: {result.get('agent_id', AGENT_ID)}\n"
        output += f"Detections: {len(result.get('detections', []))}\n"
        output += f"Area Covered: {result.get('area_covered_pct', 0):.1f}%\n"
        output += f"Mission Time: {result.get('mission_time_ms', 0)}ms\n"
        output += f"Pattern: {result.get('search_pattern', 'unknown')}\n"
        output += f"Waypoints: {result.get('waypoints_completed', 0)}/{result.get('waypoints_total', 0)}\n"
        output += f"Status: {result.get('mission_status', 'unknown')}\n"

        detections = result.get("detections", [])
        if detections:
            output += f"\nDetections:\n"
            for i, det in enumerate(detections):
                output += (
                    f"  {i+1}. {det.get('label', '?')} "
                    f"(conf={det.get('confidence', 0):.2f}) "
                    f"at ({det.get('x_cm', 0):.1f}, {det.get('y_cm', 0):.1f})cm"
                )
                if det.get("llm_match"):
                    output += " [TARGET MATCH]"
                output += "\n"

        summary = result.get("summary", "")
        if summary:
            output += f"\nSummary: {summary}\n"

        return output
