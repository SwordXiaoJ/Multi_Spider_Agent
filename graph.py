"""
LangGraph-based agent for PiCrawler.

Two graphs:
1. MissionGraph: perceive → decide → act → evaluate (Agent mode)
2. ExecuteGraph: sequential direct_control steps (Execute mode)

Each observation from Central triggers one MissionGraph invocation.
State persists across invocations via the MissionManager.
"""

import asyncio
import math
import logging
import time
import traceback
from typing import Dict, Any, Optional, List, TypedDict
from datetime import datetime

import requests as http_requests
from langgraph.graph import StateGraph, START, END

from agent_picrawler.config import AGENT_ID, PATROL_SPEED
from agent_picrawler.capabilities import ACTION_NAMES, get_actions_description
from agent_picrawler.hardware import CrawlerControl
from agent_picrawler.brain import LLMClient
from agent_picrawler import speaker

logger = logging.getLogger(__name__)

AVAILABLE_ACTIONS = ACTION_NAMES + ["MISSION_COMPLETE"]
SETUP_ACTIONS = {"stand_up", "sit_down", "stop"}
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ── State types ─────────────────────────────────────────────

class MissionState(TypedDict):
    """State for the mission graph. Passed through all nodes."""
    # Mission context
    task_goal: str
    mission_id: str
    callback_url: str
    # Observation input
    obs_status: str            # "searching", "finished", "init"
    obs_reason: str            # reason for finished (timeout, cancelled)
    detections: List[Dict]
    has_target: bool
    elapsed_s: float
    # Robot state
    position: tuple            # (x, y, heading)
    standing: bool
    # Decision output
    obs_description: str
    action: str
    reason: str
    # Result
    outcome: str               # "acted", "setup", "completed", "ignored", "error"


class ExecuteState(TypedDict):
    """State for the execute graph."""
    steps: List[Dict]
    current_step: int
    results: List[Dict]
    status: str


# ── Helper functions ────────────────────────────────────────

def _describe_bbox(bbox) -> str:
    if not bbox:
        return "unknown position"
    # Support both list [x1,y1,x2,y2] and dict {"x1":..,"y1":..,"x2":..,"y2":..}
    if isinstance(bbox, dict):
        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        return "unknown position"
    x_center = (x1 + x2) / 2
    box_height = y2 - y1
    h_pos = "left side" if x_center < FRAME_WIDTH * 0.33 else (
        "right side" if x_center > FRAME_WIDTH * 0.67 else "center")
    distance = "very close" if box_height > FRAME_HEIGHT * 0.5 else (
        "medium distance" if box_height > FRAME_HEIGHT * 0.2 else "far away")
    return f"{h_pos}, {distance}"


def _describe_detections(detections: list) -> str:
    if not detections:
        return "No objects detected."
    parts = []
    for det in detections:
        label = det.get("label", "unknown")
        conf = float(det.get("confidence", 0) or 0)
        position = _describe_bbox(det.get("bbox", []))
        desc = f"- {label} (confidence: {conf:.0%}, position: {position})"
        if det.get("is_target"):
            desc += " [TARGET]"
        parts.append(desc)
    return "\n".join(parts)


# ── Search pattern generation (inlined) ─────────────────────

def _generate_lawnmower(width_cm=60.0, height_cm=60.0, spacing_cm=30.0):
    waypoints = []
    for i in range(int(height_cm / spacing_cm) + 1):
        y = min(i * spacing_cm, height_cm)
        waypoints.append((width_cm if i % 2 == 0 else 0.0, y))
    return waypoints


def _generate_spiral(radius_cm=100.0, spacing_cm=30.0):
    waypoints = []
    for loop in range(1, int(radius_cm / spacing_cm) + 1):
        r = spacing_cm * loop
        for pt in range(12):
            angle = 2 * math.pi * pt / 12
            waypoints.append((r * math.cos(angle), r * math.sin(angle)))
    return waypoints


def _generate_expanding_square(size_cm=200.0, spacing_cm=30.0):
    waypoints = [(0.0, 0.0)]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x, y, step, half = 0.0, 0.0, 1, size_cm / 2
    while step * spacing_cm <= size_cm:
        for di in range(4):
            dx, dy = dirs[di]
            for _ in range(step if di < 2 else step + 1):
                x += dx * spacing_cm
                y += dy * spacing_cm
                if abs(x) <= half and abs(y) <= half:
                    waypoints.append((x, y))
        step += 2
    return waypoints


PATTERNS = {
    "lawnmower": _generate_lawnmower,
    "spiral": _generate_spiral,
    "expanding_square": _generate_expanding_square,
}


# ── Action execution ────────────────────────────────────────

def _build_action_map(control: CrawlerControl) -> dict:
    """Build action name → callable map from hardware control."""
    return {
        "stand_up": lambda: control.stand(),
        "sit_down": lambda: control.sit(),
        "stop": lambda: None,
        "forward": lambda: control.forward(3),
        "backward": lambda: control.backward(3),
        "turn_left": lambda: control.turn_left(5),
        "turn_right": lambda: control.turn_right(5),
        "turn_left_angle": lambda: control.turn_left_angle(5),
        "turn_right_angle": lambda: control.turn_right_angle(5),
        "wave": lambda: control.wave(2),
        "dance": lambda: control.dance(1),
        "push_up": lambda: control.push_up(2),
        "nod": lambda: control.nod(2),
        "shake_head": lambda: control.shake_head(2),
        "shake_hand": lambda: control.shake_hand(2),
        "play_dead": lambda: control.play_dead(1),
        "look_left": lambda: control.look_left(1),
        "look_right": lambda: control.look_right(1),
        "look_up": lambda: control.look_up(1),
        "look_down": lambda: control.look_down(1),
    }


# ── Mission Graph (Agent mode) ──────────────────────────────

class MissionGraph:
    """
    LangGraph-based mission processor.

    Each observation from Central triggers one graph invocation:
    START → perceive → decide → act → evaluate → END
    """

    def __init__(self, control: CrawlerControl, llm: LLMClient):
        self.control = control
        self.llm = llm
        self.action_map = _build_action_map(control)
        self.graph = self._build()

    def _build(self) -> StateGraph:
        g = StateGraph(MissionState)
        g.add_node("perceive", self._perceive)
        g.add_node("decide", self._decide)
        g.add_node("act", self._act)
        g.add_node("evaluate", self._evaluate)

        g.add_edge(START, "perceive")
        g.add_edge("perceive", "decide")
        g.add_edge("decide", "act")
        g.add_edge("act", "evaluate")
        g.add_edge("evaluate", END)

        return g.compile()

    def _perceive(self, state: MissionState) -> dict:
        """Parse observation into natural language description."""
        obs_status = state["obs_status"]
        detections = state.get("detections", [])
        has_target = state.get("has_target", False)
        reason_finished = state.get("obs_reason", "")

        if obs_status == "init":
            obs_desc = (
                "Mission just started. No observations yet. "
                "Decide what initial setup actions are needed "
                "(e.g. stand_up to prepare for detection)."
            )
        elif obs_status == "finished":
            if has_target:
                obs_desc = f"Mission ended. Final detections:\n{_describe_detections(detections)}"
            else:
                obs_desc = (
                    f"Mission ended ({reason_finished}). "
                    f"No target was found during the entire search. "
                    f"You must now execute the fallback/otherwise action from the task goal."
                )
        else:
            target_dets = [d for d in detections if d.get("is_target")]
            # For report tasks: show all detections if no targets marked
            obs_desc = _describe_detections(target_dets if target_dets else detections)

        return {"obs_description": obs_desc}

    def _decide(self, state: MissionState) -> dict:
        """Call LLM to choose the next action."""
        decision = self.llm.decide_observation_response(
            task_goal=state["task_goal"],
            observations=state["obs_description"],
            position=state["position"],
            elapsed_s=state.get("elapsed_s", 0),
            available_actions=AVAILABLE_ACTIONS,
            actions_description=get_actions_description(),
            robot_state=f"standing={state.get('standing', False)}",
        )
        action = decision.get("action", "stop")
        reason = decision.get("reason", "")
        logger.info(f"LLM decision: {action} — {reason} (obs={state['obs_status']})")
        return {"action": action, "reason": reason}

    def _act(self, state: MissionState) -> dict:
        """Execute the chosen action on the robot."""
        action = state["action"]

        if action == "MISSION_COMPLETE":
            speaker.say("Mission complete")
            return {"outcome": "completed"}

        handler = self.action_map.get(action)
        if handler:
            handler()  # synchronous hardware call
            speaker.say(action.replace("_", " "))
        else:
            logger.warning(f"Unknown action: {action}")

        return {}

    def _evaluate(self, state: MissionState) -> dict:
        """Determine the outcome of this cycle."""
        action = state["action"]
        obs_status = state["obs_status"]
        has_target = state.get("has_target", False)

        if action == "MISSION_COMPLETE":
            return {"outcome": "completed"}

        if obs_status == "finished":
            return {"outcome": "completed"}

        is_response = action not in SETUP_ACTIONS
        if has_target and is_response:
            return {"outcome": "acted"}

        if has_target and not is_response:
            logger.info(f"Setup action '{action}' with target — waiting for next observation")
            return {"outcome": "setup"}

        return {"outcome": "ignored"}

    async def run(self, state: MissionState) -> MissionState:
        """Run one cycle of the mission graph (async wrapper)."""
        return await asyncio.to_thread(self.graph.invoke, state)


# ── Mission Manager ─────────────────────────────────────────

class MissionManager:
    """
    Manages mission lifecycle and routes observations through MissionGraph.

    This replaces the old ObservationHandler + CrawlerSearchAgent.
    """

    def __init__(self, control: CrawlerControl, llm: LLMClient):
        self.control = control
        self.llm = llm
        self.mission_graph = MissionGraph(control, llm)
        self.action_map = _build_action_map(control)

        # Mission context
        self.mission_id: str | None = None
        self.task_goal: str | None = None
        self.callback_url: str | None = None
        self.mission_active = False

        # Throttle
        self._last_decision_time = 0.0
        self._min_decision_interval = 2.0
        self._processing = False

        # Auto-sit
        self._auto_sit_task: Optional[asyncio.Task] = None
        self._auto_sit_timeout = 30

        # Patrol
        self._stop_requested = False

    # ── Mission lifecycle ────────────────────────────────────

    def set_mission(self, mission_id: str, task_goal: str, callback_url: str = ""):
        self.mission_id = mission_id
        self.task_goal = task_goal
        self.callback_url = callback_url
        self.mission_active = True
        logger.info(f"Mission set: id={mission_id}, goal={task_goal}")

    def clear_mission(self):
        self.mission_id = None
        self.task_goal = None
        self.callback_url = None
        self.mission_active = False
        logger.info("Mission cleared")

    # ── Agent mode: initial actions ──────────────────────────

    async def decide_initial_actions(self):
        """Run the graph with obs_status='init' to decide setup actions."""
        if not self.task_goal:
            return

        logger.info(f"Deciding initial actions for: {self.task_goal}")
        state = self._build_state(obs_status="init")
        result = await self.mission_graph.run(state)

        action = result.get("action", "stop")
        logger.info(f"Initial action: {action} — {result.get('reason', '')}")

    # ── Agent mode: handle observation ───────────────────────

    async def handle_observation(self, data: dict) -> dict:
        """Process one observation from Central through the mission graph."""
        # Auto-activate (but not from "finished" — if mission isn't active,
        # a finished observation means we already completed it)
        if not self.mission_active:
            obs_status = data.get("status", "searching")
            if obs_status == "finished":
                logger.info("Ignoring 'finished' observation — no active mission (already completed)")
                return {"status": "ignored", "reason": "mission already completed"}
            mid = data.get("mission_id")
            goal = data.get("task_goal")
            cb = data.get("callback_url", "")
            if mid and goal:
                logger.info(f"Auto-activating mission: {mid}")
                self.set_mission(mid, goal, cb)
            else:
                return {"status": "ignored", "reason": "no active mission"}

        if data.get("callback_url") and not self.callback_url:
            self.callback_url = data["callback_url"]

        if self._processing:
            return {"status": "skipped", "reason": "already processing"}

        obs_status = data.get("status", "searching")
        detections = data.get("detections", [])
        has_target = any(d.get("is_target") for d in detections)

        report_all = data.get("report_all", False)

        # Fast path: no target and not report_all → skip
        if obs_status == "searching" and not has_target and not report_all:
            return {"status": "ignored", "reason": "no target detected"}

        # Throttle (except "finished")
        now = time.time()
        if obs_status != "finished" and now - self._last_decision_time < self._min_decision_interval:
            return {"status": "throttled", "reason": "too frequent"}

        self._processing = True
        self._last_decision_time = now

        try:
            state = self._build_state(
                obs_status=obs_status,
                obs_reason=data.get("reason", ""),
                detections=detections,
                has_target=has_target,
                elapsed_s=data.get("elapsed_s", 0),
                task_goal=data.get("task_goal", self.task_goal or ""),
            )

            result = await self.mission_graph.run(state)

            action = result.get("action", "stop")
            reason = result.get("reason", "")
            outcome = result.get("outcome", "ignored")

            if outcome == "completed" or outcome == "acted":
                if obs_status != "finished":
                    await self._notify_complete()
                else:
                    self.clear_mission()
                return {"status": "ok", "action_taken": action, "reason": reason, "mission_ended": True}

            if outcome == "setup":
                return {"status": "ok", "action_taken": action, "reason": reason}

            if obs_status == "finished":
                self.clear_mission()
                return {"status": "ok", "action_taken": action, "reason": reason, "mission_ended": True}

            return {"status": "ok", "action_taken": action, "reason": reason}

        except Exception as e:
            logger.error(f"Observation error: {e}\n{traceback.format_exc()}")
            return {"status": "error", "reason": str(e)}
        finally:
            self._processing = False

    # ── Execute mode: direct control ─────────────────────────

    async def execute_steps(self, steps: List[Dict]) -> dict:
        """Execute a list of direct_control steps sequentially."""
        results = []
        total_time = 0

        for i, step in enumerate(steps):
            action = step.get("action", "stop")
            desc = step.get("description", f"step {i+1}")
            logger.info(f"Step {i+1}/{len(steps)}: {desc}")

            self._cancel_auto_sit()
            start = datetime.utcnow()

            handler = self.action_map.get(action)
            if handler:
                await asyncio.to_thread(handler)
                speaker.say(action.replace("_", " "))
                status = "completed"
            else:
                status = "failed"
                speaker.announce_error(f"Unknown: {action}")

            elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
            total_time += elapsed
            results.append({"step": i + 1, "description": desc, "status": status})

            if action not in ("sit_down", "stop"):
                self._schedule_auto_sit()

        x, y, heading = self.control.get_position()
        summaries = [f"{r['step']}. {r['description']}: {r['status']}" for r in results]

        return {
            "agent_id": AGENT_ID,
            "detections": [],
            "area_covered_pct": 0.0,
            "mission_time_ms": total_time,
            "search_pattern": "none",
            "waypoints_completed": 0,
            "waypoints_total": 0,
            "summary": f"Executed {len(steps)} steps:\n" + "\n".join(summaries),
            "mission_status": "completed" if all(r["status"] == "completed" for r in results) else "partial",
        }

    # ── Patrol (search mode) ─────────────────────────────────

    def stop_patrol(self):
        self._stop_requested = True

    async def patrol(self, search_pattern: str = "lawnmower") -> dict:
        """Patrol along waypoints. Detection is handled by Central."""
        self._stop_requested = False
        gen = PATTERNS.get(search_pattern, _generate_lawnmower)
        waypoints = gen(width_cm=60, height_cm=60, spacing_cm=30)
        total = len(waypoints)
        start = datetime.utcnow()

        logger.info(f"Patrol: pattern={search_pattern}, waypoints={total}")
        speaker.announce_mission_start(search_pattern)

        self.control.stand()
        self.control.reset_position()
        completed = 0

        try:
            for i, (tx, ty) in enumerate(waypoints):
                if self._stop_requested:
                    logger.info("Patrol stopped")
                    break
                logger.info(f"Waypoint {i+1}/{total}: ({tx:.1f}, {ty:.1f})")
                self.control.navigate_to(tx, ty)
                completed = i + 1
        finally:
            self.control.sit()

        elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
        coverage = (completed / max(total, 1)) * 100
        speaker.announce_mission_complete(0)

        return {
            "agent_id": AGENT_ID,
            "detections": [],
            "area_covered_pct": coverage,
            "mission_time_ms": elapsed,
            "search_pattern": search_pattern,
            "waypoints_completed": completed,
            "waypoints_total": total,
            "summary": f"Patrol {search_pattern}: {completed}/{total}, {coverage:.0f}%",
            "mission_status": "completed" if completed == total else "partial",
        }

    # ── Helpers ──────────────────────────────────────────────

    def _build_state(self, obs_status="searching", **kwargs) -> MissionState:
        x, y, heading = self.control.get_position()
        return {
            "task_goal": kwargs.get("task_goal", self.task_goal or ""),
            "mission_id": self.mission_id or "",
            "callback_url": self.callback_url or "",
            "obs_status": obs_status,
            "obs_reason": kwargs.get("obs_reason", ""),
            "detections": kwargs.get("detections", []),
            "has_target": kwargs.get("has_target", False),
            "elapsed_s": kwargs.get("elapsed_s", 0),
            "position": (x, y, heading),
            "standing": self.control._standing,
            "obs_description": "",
            "action": "",
            "reason": "",
            "outcome": "",
        }

    async def _notify_complete(self):
        if not self.callback_url:
            self.clear_mission()
            return
        url = self.callback_url
        try:
            resp = await asyncio.to_thread(
                http_requests.post, url,
                json={"agent_id": AGENT_ID}, timeout=5,
            )
            logger.info(f"Mission complete: {url} → HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Complete callback failed: {e}")
        self.clear_mission()

    def _cancel_auto_sit(self):
        if self._auto_sit_task and not self._auto_sit_task.done():
            self._auto_sit_task.cancel()
            self._auto_sit_task = None

    async def _auto_sit_coroutine(self):
        try:
            await asyncio.sleep(self._auto_sit_timeout)
            logger.info(f"Auto-sit: {self._auto_sit_timeout}s idle")
            self.control.sit()
        except asyncio.CancelledError:
            pass

    def _schedule_auto_sit(self):
        self._cancel_auto_sit()
        self._auto_sit_task = asyncio.ensure_future(self._auto_sit_coroutine())
