"""
Lightweight LLM client for intelligent decision-making.

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
            "Use reasonable defaults: width_cm=200, height_cm=200, spacing_cm=30.\n"
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
                "width_cm": 200,
                "height_cm": 200,
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

    def summarize_mission(self, detections: list, stats: dict) -> str:
        """Generate a natural language mission summary."""
        system = (
            "You are a ground patrol robot. Summarize the mission results in 2-3 sentences.\n"
            "Include: area covered, objects found, key observations."
        )
        user = f"Detections: {json.dumps(detections)}\nStats: {json.dumps(stats)}"
        result = self._call(system, user)
        return result
