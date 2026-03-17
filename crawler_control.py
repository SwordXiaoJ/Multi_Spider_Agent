"""
PiCrawler movement control with dead-reckoning position tracking.

MOCK_MODE (default): only logs actions, no hardware movement.
Set MOCK_MODE=false to drive the real robot.
"""

import math
import time
import signal
import logging

from agent_picrawler.config import (
    MOCK_MODE, STEP_DISTANCE_CM, DEGREES_PER_TURN_STEP,
    OBSTACLE_THRESHOLD_CM, PATROL_SPEED,
)

logger = logging.getLogger(__name__)

# Only import hardware when not mocking
if not MOCK_MODE:
    from picrawler import Picrawler
    from robot_hat import Ultrasonic, Pin


class Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise Timeout()


class CrawlerControl:
    """
    Controls PiCrawler movement and tracks position via dead reckoning.

    Position is tracked in centimeters from the starting point.
    heading: degrees, 0 = positive Y axis (forward at start), clockwise.
    """

    def __init__(self, speed: int = PATROL_SPEED):
        self.speed = speed
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # degrees, 0 = forward (positive Y)
        self._standing = False

        if MOCK_MODE:
            logger.info("[MOCK] CrawlerControl initialized (no hardware)")
            self.crawler = None
            self.sonar = None
        else:
            self.crawler = Picrawler()
            self.sonar = Ultrasonic(Pin("D2"), Pin("D3"))
            signal.signal(signal.SIGALRM, _alarm_handler)
            logger.info("CrawlerControl initialized with hardware")

    def stand(self):
        """Stand up."""
        if MOCK_MODE:
            logger.info("[MOCK] stand()")
        else:
            self.crawler.do_action("stand", 1, self.speed)
            time.sleep(0.5)
        self._standing = True

    def sit(self):
        """Sit down."""
        if MOCK_MODE:
            logger.info("[MOCK] sit()")
        else:
            self.crawler.do_step("sit", self.speed)
            time.sleep(0.5)
        self._standing = False

    def forward(self, steps: int = 1):
        """Move forward N steps, updating position."""
        for i in range(steps):
            if MOCK_MODE:
                logger.info(f"[MOCK] forward step {i+1}/{steps}")
            else:
                self.crawler.do_action("forward", 1, self.speed)
                time.sleep(0.3)

            rad = math.radians(self.heading)
            self.x += math.sin(rad) * STEP_DISTANCE_CM
            self.y += math.cos(rad) * STEP_DISTANCE_CM

        logger.debug(f"Position after forward({steps}): ({self.x:.1f}, {self.y:.1f})")

    def backward(self, steps: int = 1):
        """Move backward N steps, updating position."""
        for i in range(steps):
            if MOCK_MODE:
                logger.info(f"[MOCK] backward step {i+1}/{steps}")
            else:
                self.crawler.do_action("backward", 1, self.speed)
                time.sleep(0.3)

            rad = math.radians(self.heading)
            self.x -= math.sin(rad) * STEP_DISTANCE_CM
            self.y -= math.cos(rad) * STEP_DISTANCE_CM

    def turn_left(self, steps: int = 1):
        """Turn left N steps, updating heading."""
        if MOCK_MODE:
            logger.info(f"[MOCK] turn_left({steps})")
        else:
            self.crawler.do_action("turn left", steps, self.speed)
            time.sleep(0.3)

        self.heading = (self.heading - DEGREES_PER_TURN_STEP * steps) % 360
        logger.debug(f"Heading after turn_left({steps}): {self.heading:.1f}°")

    def turn_right(self, steps: int = 1):
        """Turn right N steps, updating heading."""
        if MOCK_MODE:
            logger.info(f"[MOCK] turn_right({steps})")
        else:
            self.crawler.do_action("turn right", steps, self.speed)
            time.sleep(0.3)

        self.heading = (self.heading + DEGREES_PER_TURN_STEP * steps) % 360
        logger.debug(f"Heading after turn_right({steps}): {self.heading:.1f}°")

    def read_distance(self) -> float | None:
        """
        Read ultrasonic distance with median filter (from avoid.py pattern).
        Returns distance in cm, or None if read failed.
        """
        if MOCK_MODE:
            logger.debug("[MOCK] read_distance() → 100.0 (no obstacle)")
            return 100.0

        vals = []
        for _ in range(5):
            try:
                signal.alarm(1)
                d = self.sonar.read()
                signal.alarm(0)
                if d is not None and d > 0:
                    vals.append(d)
            except Timeout:
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
            time.sleep(0.03)

        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    def check_obstacle(self) -> bool:
        """Returns True if obstacle detected within threshold."""
        dist = self.read_distance()
        if dist is None:
            return False  # sensor error, assume no obstacle
        blocked = dist <= OBSTACLE_THRESHOLD_CM
        if blocked:
            logger.info(f"Obstacle detected at {dist:.1f}cm")
        return blocked

    def navigate_to(self, target_x: float, target_y: float) -> bool:
        """
        Navigate to target (x_cm, y_cm) with obstacle avoidance.

        Returns True if reached, False if blocked.
        """
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < STEP_DISTANCE_CM:
            return True

        # Calculate target heading (0° = positive Y, clockwise)
        target_heading = math.degrees(math.atan2(dx, dy)) % 360

        # Turn to face target
        self._turn_to_heading(target_heading)

        # Walk forward step by step
        steps_needed = int(distance / STEP_DISTANCE_CM)
        for i in range(steps_needed):
            if self.check_obstacle():
                logger.info(f"Obstacle at step {i+1}/{steps_needed}, attempting detour")
                if not self._detour():
                    logger.warning("Detour failed, skipping waypoint")
                    return False
                # Recalculate heading after detour
                dx = target_x - self.x
                dy = target_y - self.y
                if math.sqrt(dx * dx + dy * dy) < STEP_DISTANCE_CM:
                    return True
                target_heading = math.degrees(math.atan2(dx, dy)) % 360
                self._turn_to_heading(target_heading)

            self.forward(1)

        logger.info(f"Reached waypoint ({target_x:.1f}, {target_y:.1f}), "
                    f"actual pos ({self.x:.1f}, {self.y:.1f})")
        return True

    def _turn_to_heading(self, target_heading: float):
        """Turn to face target heading using shortest rotation."""
        diff = (target_heading - self.heading) % 360
        if diff == 0:
            return

        if diff <= 180:
            steps = round(diff / DEGREES_PER_TURN_STEP)
            if steps > 0:
                self.turn_right(steps)
        else:
            steps = round((360 - diff) / DEGREES_PER_TURN_STEP)
            if steps > 0:
                self.turn_left(steps)

    def _detour(self) -> bool:
        """Simple detour: turn left, walk a few steps, turn back."""
        self.turn_left(5)   # ~90 degrees
        self.forward(3)     # sidestep
        self.turn_right(5)  # face original direction
        self.forward(3)     # pass the obstacle
        return True

    def get_position(self) -> tuple:
        """Returns (x_cm, y_cm, heading_deg)."""
        return (self.x, self.y, self.heading)

    def reset_position(self):
        """Reset dead reckoning to origin."""
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
