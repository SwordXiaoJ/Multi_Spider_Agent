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
    LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY, MOCK_MODE,
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
        if MOCK_MODE or not self.api_key:
            logger.info(f"[MOCK] LLM call (no API key or mock mode)")
            logger.info(f"[MOCK] System: {system_prompt[:100]}...")
            logger.info(f"[MOCK] User: {user_prompt[:100]}...")
            return '{"mock": true}'

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
                "max_tokens": 1024,
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

    def parse_mission(self, command: str) -> dict:
        """
        Parse natural language mission command into structured params.

        Returns: {"pattern": "lawnmower", "target": "person", "width_cm": 200, ...}
        """
        system = (
            "You are a ground patrol robot agent. Parse the mission command into JSON.\n"
            "Return JSON with fields: pattern (lawnmower/spiral/expanding_square), "
            "target (what to look for), width_cm, height_cm, spacing_cm.\n"
            "Use reasonable defaults: width_cm=60, height_cm=60, spacing_cm=30.\n"
            "Only return valid JSON, no other text."
        )
        result = self._call(system, command)

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {result}")
            return {
                "pattern": "lawnmower",
                "target": "any",
                "width_cm": 60,
                "height_cm": 60,
                "spacing_cm": 30,
            }

    def evaluate_detection(self, target: str, detection: dict) -> dict:
        """
        Judge whether a detection matches the mission target.

        Returns: {"match": True/False, "confidence": 0.0-1.0, "reason": "..."}
        """
        system = (
            "You are a ground patrol robot. Evaluate whether the detection matches the target.\n"
            "Return JSON: {\"match\": bool, \"confidence\": float, \"reason\": \"...\"}\n"
            "Only return valid JSON."
        )
        user = f"Target: {target}\nDetection: {json.dumps(detection)}"
        result = self._call(system, user)

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"match": False, "confidence": 0.0, "reason": "parse error"}

    def decide_action(self, situation: str, options: list) -> dict:
        """
        Decide what action to take in a complex situation.

        Returns: {"action": "...", "reason": "..."}
        """
        system = (
            "You are a ground patrol robot. Given a situation, choose the best action.\n"
            "Return JSON: {\"action\": \"chosen_option\", \"reason\": \"...\"}\n"
            "Only return valid JSON."
        )
        user = f"Situation: {situation}\nOptions: {json.dumps(options)}"
        result = self._call(system, user)

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"action": options[0] if options else "wait", "reason": "parse error"}

    def decide_detection_response(
        self, detection_status: dict, position: tuple, mission_context: str
    ) -> dict:
        """
        Decide how to respond to a mid-patrol detection event.

        Returns: {"action": "ignore|approach|report", "reason": "..."}
        """
        system = (
            "You are a ground patrol robot. A detection occurred while you were moving.\n"
            "Given the detection data, your position, and mission context, choose one action:\n"
            "- ignore: false positive or irrelevant to mission target\n"
            "- approach: move closer for better identification\n"
            "- report: important finding, log it and continue patrol\n"
            "Return JSON: {\"action\": \"...\", \"reason\": \"...\"}\n"
            "Only return valid JSON."
        )
        x, y, heading = position
        user = (
            f"Detection: {json.dumps(detection_status)}\n"
            f"Position: ({x:.1f}, {y:.1f}), heading: {heading:.1f}°\n"
            f"Mission: {mission_context}"
        )
        result = self._call(system, user)

        try:
            parsed = json.loads(result)
            if parsed.get("action") not in ("ignore", "approach", "report"):
                parsed["action"] = "approach"
            return parsed
        except json.JSONDecodeError:
            return {"action": "approach", "reason": "parse error, defaulting to approach"}

    def decide_observation_response(
        self,
        task_goal: str,
        observations: str,
        position: tuple,
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
            "- If a [TARGET] is at center and far, move forward\n"
            "- If a [TARGET] is at center and close, execute the response action "
            "specified in the task goal (e.g. wave, dance, sit_down)\n"
            "- look_left/look_right only tilts the body, does NOT change facing direction\n"
            "- turn_left/turn_right changes facing direction (~90 degrees)\n"
            "- If the task goal is fully accomplished, output MISSION_COMPLETE\n"
            "- IMPORTANT: If the task goal only asks to 'detect', 'report', 'observe', "
            "'identify', or 'describe' (no physical action like wave/dance/sit), "
            "output MISSION_COMPLETE immediately after seeing the detections. "
            "Do NOT perform any physical action — reporting is automatic.\n"
            "- Only perform physical actions (wave, dance, sit_down, etc.) if the task goal "
            "explicitly asks for them.\n"
            "- If you are already standing (standing=True), do NOT choose stand_up again\n"
            "- If no targets detected, continue patrol (do nothing — return stop)\n\n"
            "Return JSON: {\"action\": \"...\", \"reason\": \"brief explanation\"}\n"
            "Only return valid JSON."
        )
        x, y, heading = position
        user = (
            f"Mission goal: {task_goal}\n"
            f"My position: ({x:.1f}, {y:.1f}) cm, heading: {heading:.1f}°\n"
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

    def summarize_mission(self, detections: list, stats: dict) -> str:
        """Generate a natural language mission summary."""
        system = (
            "You are a ground patrol robot. Summarize the mission results in 2-3 sentences.\n"
            "Include: area covered, objects found, key observations."
        )
        user = f"Detections: {json.dumps(detections)}\nStats: {json.dumps(stats)}"
        result = self._call(system, user)
        return result
