"""
PiCrawler search agent using LangGraph workflow.

Workflow: START -> plan_mission -> execute_mission -> analyze_results -> finalize -> END
"""

import logging
import os
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from agent_picrawler.config import AGENT_ID, PATROL_SPEED
from agent_picrawler.crawler_control import CrawlerControl
from agent_picrawler.crawler_camera import CrawlerCamera
from agent_picrawler.llm_client import LLMClient
from agent_picrawler.mission_executor import MissionExecutor

logger = logging.getLogger(__name__)


class CrawlerSearchState(TypedDict):
    """State maintained throughout the search workflow."""
    request_id: str
    prompt: str
    target_type: str
    mission_config: Optional[Dict[str, Any]]
    detections: List[Dict[str, Any]]
    mission_status: str
    area_covered_pct: float
    mission_time_ms: int
    waypoints_completed: int
    waypoints_total: int
    search_pattern: str
    summary: str
    error: str
    messages: Annotated[list, add_messages]


class CrawlerSearchAgent:
    """
    PiCrawler search agent for ground patrol.

    Uses LangGraph workflow: plan -> execute -> analyze -> finalize.
    """

    def __init__(self, agent_id: str = AGENT_ID):
        self.agent_id = agent_id

        self.control = CrawlerControl(speed=PATROL_SPEED)
        self.camera = CrawlerCamera()
        self.llm = LLMClient()
        self.executor = MissionExecutor(self.control, self.camera, self.llm)

        self.graph = self._build_graph()
        logger.info(f"CrawlerSearchAgent initialized: {agent_id}")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CrawlerSearchState)

        workflow.add_node("plan_mission", self._plan_mission_node)
        workflow.add_node("execute_mission", self._execute_mission_node)
        workflow.add_node("analyze_results", self._analyze_results_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.add_edge(START, "plan_mission")
        workflow.add_edge("plan_mission", "execute_mission")
        workflow.add_edge("execute_mission", "analyze_results")
        workflow.add_edge("analyze_results", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _plan_mission_node(self, state: CrawlerSearchState) -> dict:
        """Use LLM to parse the prompt into a mission config."""
        prompt = state.get("prompt", "patrol this area")
        target = state.get("target_type", "any")
        override_pattern = state.get("search_pattern", "")

        mission_config = self.llm.parse_mission(prompt)
        mission_config["target"] = target
        mission_config.setdefault("pattern", "lawnmower")
        mission_config.setdefault("detect_face", True)

        # Metadata pattern overrides LLM
        if override_pattern:
            mission_config["pattern"] = override_pattern

        pattern = mission_config["pattern"]
        logger.info(f"Mission planned: pattern={pattern}, target={target}")

        return {
            "mission_config": mission_config,
            "search_pattern": pattern,
            "messages": [AIMessage(content=f"Mission planned: {pattern} pattern, target={target}")],
        }

    async def _execute_mission_node(self, state: CrawlerSearchState) -> dict:
        """Execute the patrol mission."""
        config = state.get("mission_config", {})

        try:
            result = self.executor.execute(config)

            return {
                "detections": result.get("detections", []),
                "area_covered_pct": result.get("area_covered_pct", 0.0),
                "mission_time_ms": result.get("mission_time_ms", 0),
                "waypoints_completed": result.get("waypoints_completed", 0),
                "waypoints_total": result.get("waypoints_total", 0),
                "summary": result.get("summary", ""),
                "mission_status": "completed",
                "messages": [AIMessage(
                    content=f"Mission completed: {len(result.get('detections', []))} detections, "
                            f"{result.get('area_covered_pct', 0):.0f}% covered"
                )],
            }
        except Exception as e:
            logger.error(f"Mission execution failed: {e}")
            return {
                "detections": [],
                "mission_status": "failed",
                "error": str(e),
                "messages": [AIMessage(content=f"Mission failed: {e}")],
            }

    async def _analyze_results_node(self, state: CrawlerSearchState) -> dict:
        """Analyze detection results."""
        detections = state.get("detections", [])
        target = state.get("target_type", "any")

        matched = [d for d in detections if d.get("llm_match", False)]

        if matched:
            msg = f"Analysis: {len(matched)} target matches out of {len(detections)} detections"
        elif detections:
            msg = f"Analysis: {len(detections)} objects detected but none matched target '{target}'"
        else:
            msg = f"Analysis: No objects detected during patrol"

        return {
            "messages": [AIMessage(content=msg)],
        }

    async def _finalize_node(self, state: CrawlerSearchState) -> dict:
        """Finalize the search result."""
        status = state.get("mission_status", "unknown")
        summary = state.get("summary", "")
        return {
            "messages": [AIMessage(content=f"Patrol finalized: {status}. {summary}")],
        }

    async def search(self, request: Dict[str, Any]) -> dict:
        """Main entry point — runs the LangGraph patrol workflow."""
        request_id = request.get("request_id", f"req-{datetime.utcnow().timestamp()}")
        prompt = request.get("prompt", "Patrol this area")
        target_type = request.get("target_type", "any")

        search_pattern = request.get("search_pattern", "")

        initial_state: CrawlerSearchState = {
            "request_id": request_id,
            "prompt": prompt,
            "target_type": target_type,
            "mission_config": None,
            "detections": [],
            "mission_status": "pending",
            "area_covered_pct": 0.0,
            "mission_time_ms": 0,
            "waypoints_completed": 0,
            "waypoints_total": 0,
            "search_pattern": search_pattern,
            "summary": "",
            "error": "",
            "messages": [HumanMessage(content=prompt)],
        }

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "request_id": request_id,
            "agent_id": self.agent_id,
            "detections": final_state.get("detections", []),
            "area_covered_pct": final_state.get("area_covered_pct", 0.0),
            "mission_time_ms": final_state.get("mission_time_ms", 0),
            "search_pattern": final_state.get("search_pattern", "lawnmower"),
            "waypoints_completed": final_state.get("waypoints_completed", 0),
            "waypoints_total": final_state.get("waypoints_total", 0),
            "summary": final_state.get("summary", ""),
            "mission_status": final_state.get("mission_status", "unknown"),
        }
