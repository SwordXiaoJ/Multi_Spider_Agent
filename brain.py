"""
Brain: LLM-powered decision engine for the PiCrawler agent.

All agent intelligence lives here — mission parsing, detection evaluation,
observation response, and mission summarization.

Supports OpenAI and Anthropic via raw HTTP (no heavy SDK needed).
Model format: "provider/model_name" (e.g. "openai/gpt-4o-mini")
"""

import json
import logging

import requests

from agent_picrawler.config import (
    LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
)

logger = logging.getLogger(__name__)


def _parse_provider(model_str: str) -> tuple:
    """Parse 'provider/model' into (provider, model_name)."""
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
        return provider.lower(), model_name
    return "openai", model_str


class LLMClient:
    """Calls cloud LLM API for agent decision-making."""

    def __init__(self, model: str = LLM_MODEL):
        self.provider, self.model = _parse_provider(model)

        if self.provider == "openai":
            self.api_key = OPENAI_API_KEY
            self.base_url = "https://api.openai.com"
        elif self.provider == "anthropic":
            self.api_key = ANTHROPIC_API_KEY
            self.base_url = "https://api.anthropic.com"
        else:
            self.api_key = OPENAI_API_KEY
            self.base_url = "https://api.openai.com"

        logger.info(f"LLMClient: provider={self.provider}, model={self.model}")

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single LLM API call. Returns response text."""
        if not self.api_key:
            logger.warning("LLM call skipped: no API key configured")
            return '{"error": "no API key"}'

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt)
            else:
                return self._call_openai(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return '{"error": "LLM call failed"}'

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_completion_tokens": 1024,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def decompose_actions(self, prompt: str, available_actions: list) -> list:
        """
        Parse a natural language command into an ordered list of robot actions.

        Called when Central sends task_type="direct_control" with a free-text prompt.
        The LLM decomposes the prompt into a sequence of executable actions.

        Returns: list of action name strings, e.g. ["wave", "dance"]
        """
        system = (
            "You are a PiCrawler robot controller. Parse the user command into "
            "an ordered list of actions. Available actions:\n"
            f"{', '.join(available_actions)}\n\n"
            "Rules:\n"
            "- Only use actions from the list above\n"
            "- The robot must stand_up before it can move or perform gestures. "
            "If the command implies movement or gestures and doesn't start with stand_up, "
            "prepend stand_up automatically.\n"
            "- Output JSON: {\"actions\": [\"action1\", \"action2\", ...]}\n"
            "- Only return valid JSON, nothing else."
        )
        user = prompt

        result = self._call(system, user)

        try:
            parsed = json.loads(result)
            actions = parsed.get("actions", [])
            # Validate each action
            valid = [a for a in actions if a in available_actions]
            if not valid:
                logger.warning(f"No valid actions from LLM decomposition: {actions}")
                return ["stop"]
            logger.info(f"LLM decomposed '{prompt}' → {valid}")
            return valid
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM decomposition: {result}")
            return ["stop"]

    def decide_observation_response(
        self,
        task_goal: str,
        observations: str,
        elapsed_s: float,
        available_actions: list,
        actions_description: str = "",
        robot_state: str = "",
    ) -> dict:
        """
        Decide what to do based on Central's observation push.

        This is the core agent decision: Pi sees what Central detected,
        and autonomously decides the next action.

        Returns: {"action": "...", "reason": "..."}
        """
        system = (
            "You are a PiCrawler robot agent. You are a spider-like quadruped robot "
            "with 4 legs and a camera. You receive observations from a camera system "
            "and must decide your next action to accomplish your mission.\n\n"
            "Your available actions:\n"
            f"{actions_description}\n\n"
            "Rules:\n"
            "- Choose exactly ONE action from the list above\n"
            "- You must stand_up before you can move or perform gestures\n"
            "- If a [TARGET] is on the left, turn_left to face it\n"
            "- If a [TARGET] is on the right, turn_right to face it\n"
            "- If a [TARGET] is at center and far, move forward to approach\n"
            "- If a [TARGET] is at center and close (or ultrasonic ≤ 30cm), "
            "execute the response action specified in the task goal (e.g. wave, dance, sit_down)\n"
            "- Ultrasonic sensor measures distance to nearest object ahead (in cm). "
            "Use it together with bbox size to judge distance. ≤30cm = close, >30cm = far\n"
            "- look_left/look_right only tilts the body, does NOT change facing direction\n"
            "- turn_left/turn_right changes facing direction (~90 degrees)\n"
            "- IMPORTANT: There are two types of tasks:\n"
            "  1) REPORT tasks ('report', 'detect', 'observe', 'identify', 'describe'): "
            "Do NOT output MISSION_COMPLETE. Just output 'stop' and let the patrol continue. "
            "Reporting is automatic — the system records all detections along the route.\n"
            "  2) ACTION tasks ('if found X, wave/dance/sit_down', etc.): "
            "When the [TARGET] is found and close, execute the specified action, "
            "then output MISSION_COMPLETE to stop the patrol.\n"
            "- Only perform physical actions (wave, dance, sit_down, etc.) if the task goal "
            "explicitly asks for them.\n"
            "- If you are already standing (standing=True), do NOT choose stand_up again\n"
            "- If no targets detected, continue patrol (do nothing — return stop)\n\n"
            "Return JSON: {\"action\": \"...\", \"reason\": \"brief explanation\"}\n"
            "Only return valid JSON."
        )
        user = (
            f"Mission goal: {task_goal}\n"
            f"Current state: {robot_state}\n"
            f"Time elapsed: {elapsed_s:.0f}s\n\n"
            f"Observations:\n{observations}\n\n"
            f"Available actions: {', '.join(available_actions)}"
        )
        result = self._call(system, user)

        try:
            parsed = json.loads(result)
            action = parsed.get("action", "stop")
            if action not in available_actions:
                logger.warning(f"LLM chose invalid action '{action}', defaulting to stop")
                parsed["action"] = "stop"
            return parsed
        except json.JSONDecodeError:
            return {"action": "stop", "reason": "LLM parse error"}

