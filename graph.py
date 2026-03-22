"""
LangGraph-based agent for PiCrawler.

Two graphs:
1. MissionGraph: perceive → decide → act → evaluate (Agent mode)
2. ExecuteGraph: sequential direct_control steps (Execute mode)

Each observation from Central triggers one MissionGraph invocation.
State persists across invocations via the MissionManager.
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, Optional, List, TypedDict
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
# Navigation actions position the robot but don't fulfill the mission goal
NAVIGATION_ACTIONS = {
    "forward", "backward", "turn_left", "turn_right",
    "turn_left_angle", "turn_right_angle",
    "look_left", "look_right", "look_up", "look_down",
}
# Gesture actions that fulfill a mission goal (wave at apple, dance, etc.)
GESTURE_ACTIONS = {
    "wave", "dance", "push_up", "nod", "shake_head",
    "shake_hand", "play_dead",
}
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
    h_pos = "left side" if x_center < FRAME_WIDTH * 0.25 else (
        "right side" if x_center > FRAME_WIDTH * 0.75 else "center")
    distance = "very close" if box_height > FRAME_HEIGHT * 0.5 else (
        "medium distance" if box_height > FRAME_HEIGHT * 0.15 else "far away")
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


# ── Target position estimation ─────────────────────────────
# Pi Camera v2 horizontal FOV ≈ 62°

CAMERA_H_FOV = 62.0

def _get_bbox_center_x(bbox) -> float | None:
    """Get horizontal center of bbox in pixels."""
    if isinstance(bbox, dict):
        x1, x2 = bbox.get("x1", 0), bbox.get("x2", 0)
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, x2 = bbox[0], bbox[2]
    else:
        return None
    return (x1 + x2) / 2


def _get_target_horizontal_position(bbox) -> str:
    """Determine if target is left, right, or center.
    Uses wide center zone (25%-75%) so robot doesn't need to aim precisely."""
    cx = _get_bbox_center_x(bbox)
    if cx is None:
        return "center"
    if cx < FRAME_WIDTH * 0.25:
        return "left"
    elif cx > FRAME_WIDTH * 0.75:
        return "right"
    return "center"


def _estimate_turn_degrees(bbox) -> float:
    """Estimate how many degrees to turn to center the target."""
    cx = _get_bbox_center_x(bbox)
    if cx is None:
        return 15.0  # default guess
    # Pixel offset from frame center
    pixel_offset = abs(cx - FRAME_WIDTH / 2)
    # Convert to degrees: pixels / half_width * half_FOV
    degrees = pixel_offset / (FRAME_WIDTH / 2) * (CAMERA_H_FOV / 2)
    return round(max(degrees, 5.0), 1)


def _extract_gesture_from_goal(task_goal: str) -> str | None:
    """
    Extract gesture action keyword from task goal string.
    e.g. "if detect an apple, wave" → "wave"
         "find a person and dance" → "dance"
    """
    goal_lower = task_goal.lower()
    for action in GESTURE_ACTIONS:
        # Match action name with word boundaries (e.g. "wave" not "waved")
        keyword = action.replace("_", " ")
        if keyword in goal_lower or action in goal_lower:
            return action
    return None


# ── Action execution ────────────────────────────────────────

def _build_action_map(control: CrawlerControl) -> dict:
    """Build action name → callable map from hardware control."""
    return {
        "stand_up": lambda: control.stand(),
        "sit_down": lambda: control.sit(),
        "stop": lambda: None,
        "forward": lambda: control.forward(3),
        "backward": lambda: control.backward(3),
        "turn_left": lambda: control.turn_left(2),       # ~76° (38°/step)
        "turn_right": lambda: control.turn_right(2),     # ~76° (38°/step)
        "turn_left_angle": lambda: control.turn_left_angle(2),   # ~26° (angle=30 default)
        "turn_right_angle": lambda: control.turn_right_angle(2), # ~26° (angle=30 default)
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

        # Only gesture/response actions (wave, dance, etc.) complete the mission
        # Navigation actions (turn, forward, look) are just positioning
        if action in SETUP_ACTIONS or action in NAVIGATION_ACTIONS:
            if has_target:
                logger.info(f"Positioning action '{action}' with target — waiting for next observation")
            return {"outcome": "setup" if has_target else "ignored"}

        # Gesture/response action with target = mission fulfilled
        if has_target:
            return {"outcome": "acted"}

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
        self.report_all = False  # True = report task (don't interrupt patrol)

        # Throttle
        self._last_decision_time = 0.0
        self._min_decision_interval = 2.0
        self._processing = False

        # Auto-sit
        self._auto_sit_task: Optional[asyncio.Task] = None
        self._auto_sit_timeout = 30

        # Patrol
        self._stop_requested = False
        self._patrol_active = False

        # Track completed missions to prevent re-activation
        self._completed_mission_ids: set = set()

        # Track consecutive turns to detect oscillation
        self._consecutive_turns = 0

    # ── Mission lifecycle ────────────────────────────────────

    def set_mission(self, mission_id: str, task_goal: str, callback_url: str = "", report_all: bool = False):
        self.mission_id = mission_id
        self.task_goal = task_goal
        self.callback_url = callback_url
        self.mission_active = True
        self.report_all = report_all
        logger.info(f"Mission set: id={mission_id}, goal={task_goal}, report_all={report_all}")

    def clear_mission(self):
        self.mission_id = None
        self.task_goal = None
        self.callback_url = None
        self.mission_active = False
        self.report_all = False
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
            if mid and mid in self._completed_mission_ids:
                logger.info(f"Ignoring observation for already-completed mission: {mid}")
                return {"status": "ignored", "reason": "mission already completed"}
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

        # Report task: just log detections, no LLM needed, patrol continues
        if self.report_all and obs_status == "searching":
            labels = [d.get("label", "?") for d in detections]
            logger.info(f"Report task: detected {labels}, patrol continues")
            return {"status": "ok", "action_taken": "stop", "reason": f"logged: {labels}"}

        # Throttle (except "finished")
        now = time.time()
        if obs_status != "finished" and now - self._last_decision_time < self._min_decision_interval:
            return {"status": "throttled", "reason": "too frequent"}

        self._processing = True
        self._last_decision_time = now

        try:
            # ── Action task with target during patrol ──
            if has_target and not self.report_all and self._patrol_active:
                return await self._handle_target_during_patrol(data, detections)

            # ── Action task with target (no patrol running) — LLM decides ──
            if has_target and not self.report_all:
                logger.info("Target found — LLM deciding action...")
                return await self._llm_decide_and_execute(data, detections)

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
                if self.report_all:
                    logger.info(f"Report task: detection logged, patrol continues")
                    return {"status": "ok", "action_taken": "stop", "reason": reason}
                else:
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

    async def _handle_target_during_patrol(self, data: dict, detections: list) -> dict:
        """
        Handle target detection while patrol is running.
        Stops patrol and lets LLM take over via normal observation flow.
        """
        logger.info("Target detected during patrol — stopping patrol, LLM takes over")
        self._consecutive_turns = 0
        self.stop_patrol()
        for _ in range(30):  # wait up to 3s for patrol to stop
            if not self._patrol_active:
                break
            await asyncio.sleep(0.1)

        # Fall through to normal LLM decision path
        return await self._llm_decide_and_execute(data, detections)

    async def _llm_decide_and_execute(self, data: dict, detections: list) -> dict:
        """
        Ask LLM to decide next action for target interaction.

        LLM sees target position (left/right/center) and distance (far/close),
        then decides: turn toward it, move closer, or execute response gesture.
        """
        task_goal = data.get("task_goal", self.task_goal or "")
        obs_desc = _describe_detections(
            [d for d in detections if d.get("is_target")] or detections
        )

        # Add ultrasonic distance reading
        ultrasonic_cm = self.control.read_distance()
        if ultrasonic_cm is not None:
            obs_desc += f"\n\nUltrasonic sensor: obstacle/object at {ultrasonic_cm:.0f}cm ahead"
        else:
            obs_desc += "\n\nUltrasonic sensor: no reading"

        # After too many turns, hint LLM to stop turning and act
        if self._consecutive_turns >= 3:
            obs_desc += (
                "\n\nIMPORTANT: You have already turned multiple times. "
                "The target is approximately in front of you. "
                "Do NOT turn again. Either move forward to approach, "
                "or execute the response gesture (e.g. wave) now."
            )

        # Ask LLM (up to 2 attempts, guard against stand_up when standing)
        for attempt in range(2):
            decision = await asyncio.to_thread(
                self.llm.decide_observation_response,
                task_goal=task_goal,
                observations=obs_desc,
                elapsed_s=data.get("elapsed_s", 0),
                available_actions=AVAILABLE_ACTIONS,
                actions_description=get_actions_description(),
                robot_state=f"standing={self.control._standing}",
            )
            action = decision.get("action", "stop")
            reason = decision.get("reason", "")
            logger.info(f"LLM decision (attempt {attempt+1}): {action} — {reason}")

            if action == "stand_up" and self.control._standing:
                logger.info("LLM said stand_up but already standing — retrying")
                continue
            break

        # Gesture action → execute and complete mission
        if action in GESTURE_ACTIONS or action == "MISSION_COMPLETE":
            self._consecutive_turns = 0
            if action != "MISSION_COMPLETE":
                handler = self.action_map.get(action)
                if handler:
                    await asyncio.to_thread(handler)
                    speaker.say(action.replace("_", " "))
            speaker.say("Mission complete")
            await self._notify_complete()
            return {"status": "ok", "action_taken": action, "reason": reason, "mission_ended": True}

        # Turn action → use bbox to calculate precise angle
        if action in ("turn_left", "turn_right", "turn_left_angle", "turn_right_angle"):
            self._consecutive_turns += 1
            target_dets = [d for d in detections if d.get("is_target")]
            bbox = target_dets[0].get("bbox", {}) if target_dets else {}
            turn_degrees = _estimate_turn_degrees(bbox)
            direction = "left" if "left" in action else "right"
            logger.info(f"LLM said {action} — turning {direction} ~{turn_degrees}° (from bbox) [turn #{self._consecutive_turns}]")
            await asyncio.to_thread(self._turn_by_degrees, direction, turn_degrees)
            return {"status": "ok", "action_taken": action, "reason": f"turned {direction} ~{turn_degrees}°"}

        # Other navigation (forward, backward, etc.) → execute directly
        if action in NAVIGATION_ACTIONS:
            self._consecutive_turns = 0
            handler = self.action_map.get(action)
            if handler:
                logger.info(f"Executing navigation: {action}")
                await asyncio.to_thread(handler)
            return {"status": "ok", "action_taken": action, "reason": reason}

        # stand_up — execute it so _standing becomes True, then wait
        if action == "stand_up":
            logger.info("Executing stand_up, then waiting for next observation")
            await asyncio.to_thread(self.control.stand)
            return {"status": "ok", "action_taken": action, "reason": reason}

        # LLM returned stop or other — wait for next observation
        logger.info(f"LLM suggested '{action}' — waiting for next observation")
        return {"status": "ok", "action_taken": action, "reason": reason}

    def _turn_by_degrees(self, direction: str, degrees: float):
        """Turn by estimated degrees using turn_left_angle/turn_right_angle.
        Calibration: actual turn ≈ angle_param × 0.55 per step.
        (tuned down from 0.44 to reduce overshoot)"""
        if degrees <= 45:
            angle_param = int(degrees / 0.55)
            steps = 1
        else:
            angle_param = int(degrees / 0.55 / 2)
            steps = 2
        angle_param = max(10, min(angle_param, 90))

        if direction == "left":
            self.control.turn_left_angle(steps, angle=angle_param)
        else:
            self.control.turn_right_angle(steps, angle=angle_param)

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

        summaries = [f"{r['step']}. {r['description']}: {r['status']}" for r in results]

        return {
            "agent_id": AGENT_ID,
            "detections": [],
            "mission_time_ms": total_time,
            "summary": f"Executed {len(steps)} steps:\n" + "\n".join(summaries),
            "mission_status": "completed" if all(r["status"] == "completed" for r in results) else "partial",
        }

    # ── Patrol (search mode) ─────────────────────────────────

    def stop_patrol(self):
        self._stop_requested = True

    async def patrol(self) -> dict:
        """Patrol a fixed route with LLM-assisted obstacle avoidance.
        Detection is handled by Central; obstacle avoidance asks LLM."""
        self._stop_requested = False
        self._patrol_active = True
        start = datetime.utcnow()

        logger.info("Patrol: fixed route mode")
        speaker.announce_mission_start("patrol")

        try:
            if not self.control._standing:
                await asyncio.to_thread(self.control.stand)

            route = self.control.PATROL_ROUTE
            logger.info(f"patrol: started, {len(route)} steps")

            for i, (action, count) in enumerate(route):
                if self._stop_requested:
                    logger.info(f"patrol: aborted at step {i+1}")
                    break

                logger.info(f"patrol: step {i+1}/{len(route)} — {action}({count})")

                if action == "forward":
                    await self._patrol_forward(count)
                elif action == "backward":
                    await asyncio.to_thread(self.control.backward, count)
                elif action == "turn_left":
                    await asyncio.to_thread(self.control.turn_left_angle, 2, count)
                elif action == "turn_right":
                    await asyncio.to_thread(self.control.turn_right_angle, 2, count)

            if not self._stop_requested:
                logger.info("patrol: route completed")
        finally:
            self._patrol_active = False
            # Only sit down if patrol finished normally (not interrupted for target)
            if not self._stop_requested:
                self.control.sit()

        elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
        speaker.announce_mission_complete(0)

        # Report task: route finished, notify Central
        if self.report_all and self.mission_active:
            await self._notify_complete()

        return {
            "agent_id": AGENT_ID,
            "detections": [],
            "mission_time_ms": elapsed,
            "summary": f"Patrol completed ({elapsed // 1000}s)",
            "mission_status": "completed",
        }

    async def _patrol_forward(self, steps: int):
        """Walk forward N steps with LLM-assisted obstacle avoidance."""
        walked = 0
        while walked < steps:
            if self._stop_requested:
                return

            obstacle = await asyncio.to_thread(self.control.check_obstacle)
            if obstacle:
                if self._stop_requested:
                    logger.info("patrol: obstacle detected but stop requested (target found), skipping avoidance")
                    return
                dist = await asyncio.to_thread(self.control.read_distance)
                dist_cm = dist if dist is not None else 0
                logger.info(f"patrol: obstacle at {dist_cm:.0f}cm, step {walked+1}/{steps} — asking LLM")
                await self._llm_decide_obstacle(dist_cm)
                if self._stop_requested:
                    return
                # After LLM avoidance action, re-check before continuing
            else:
                await asyncio.to_thread(self.control.forward, 1)
                walked += 1

    async def _llm_decide_obstacle(self, distance_cm: float):
        """Ask LLM how to handle an obstacle during patrol."""
        system = (
            "You are a PiCrawler robot patrolling an area. "
            "An obstacle has been detected ahead by the ultrasonic sensor.\n\n"
            "You must choose ONE avoidance action:\n"
            "- backward: back up a few steps\n"
            "- turn_left: turn left to go around the obstacle\n"
            "- turn_right: turn right to go around the obstacle\n"
            "- stop: stop and wait\n\n"
            "Consider the distance. If very close (<10cm), back up first. "
            "Otherwise, turn to go around it.\n\n"
            "Return JSON: {\"action\": \"...\", \"reason\": \"brief explanation\"}\n"
            "Only return valid JSON."
        )
        user = f"Obstacle detected at {distance_cm:.0f}cm ahead. What should I do?"

        result = await asyncio.to_thread(self.llm._call, system, user)
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, Exception):
            parsed = {"action": "turn_left", "reason": "parse error, default avoidance"}

        action = parsed.get("action", "turn_left")
        reason = parsed.get("reason", "")
        logger.info(f"LLM obstacle decision: {action} — {reason}")

        if action == "backward":
            await asyncio.to_thread(self.control.backward, 2)
        elif action == "turn_left":
            await asyncio.to_thread(self.control.turn_left_angle, 2, 45)
            await asyncio.to_thread(self.control.forward, 2)
        elif action == "turn_right":
            await asyncio.to_thread(self.control.turn_right_angle, 2, 45)
            await asyncio.to_thread(self.control.forward, 2)
        else:
            await asyncio.to_thread(self.control.backward, 1)

    # ── Helpers ──────────────────────────────────────────────

    def _build_state(self, obs_status="searching", **kwargs) -> MissionState:
        return {
            "task_goal": kwargs.get("task_goal", self.task_goal or ""),
            "mission_id": self.mission_id or "",
            "callback_url": self.callback_url or "",
            "obs_status": obs_status,
            "obs_reason": kwargs.get("obs_reason", ""),
            "detections": kwargs.get("detections", []),
            "has_target": kwargs.get("has_target", False),
            "elapsed_s": kwargs.get("elapsed_s", 0),
            "standing": self.control._standing,
            "obs_description": "",
            "action": "",
            "reason": "",
            "outcome": "",
        }

    async def _notify_complete(self):
        # Record completed mission to prevent re-activation
        if self.mission_id:
            self._completed_mission_ids.add(self.mission_id)
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
