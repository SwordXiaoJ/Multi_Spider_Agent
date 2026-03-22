"""
Hardware driver: PiCrawler servo control with ultrasonic obstacle avoidance.

Talks directly to PiCrawler servos and ultrasonic sensor.
MOCK_MODE (default): only logs actions, no hardware movement.
Set MOCK_MODE=false to drive the real robot.
"""

import time
import signal
import logging

from agent_picrawler.config import (
    MOCK_MODE, OBSTACLE_THRESHOLD_CM, PATROL_SPEED,
)
from agent_picrawler import speaker

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
    Controls PiCrawler movement with ultrasonic obstacle avoidance.
    """

    def __init__(self, speed: int = PATROL_SPEED):
        self.speed = speed
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
        """Move forward N steps."""
        for i in range(steps):
            if MOCK_MODE:
                logger.info(f"[MOCK] forward step {i+1}/{steps}")
            else:
                self.crawler.do_action("forward", 1, self.speed)
                time.sleep(0.3)

    def backward(self, steps: int = 1):
        """Move backward N steps."""
        for i in range(steps):
            if MOCK_MODE:
                logger.info(f"[MOCK] backward step {i+1}/{steps}")
            else:
                self.crawler.do_action("backward", 1, self.speed)
                time.sleep(0.3)

    def turn_left(self, steps: int = 1):
        """Turn left N steps."""
        if MOCK_MODE:
            logger.info(f"[MOCK] turn_left({steps})")
        else:
            self.crawler.do_action("turn left", steps, self.speed)
            time.sleep(0.3)

    def turn_right(self, steps: int = 1):
        """Turn right N steps."""
        if MOCK_MODE:
            logger.info(f"[MOCK] turn_right({steps})")
        else:
            self.crawler.do_action("turn right", steps, self.speed)
            time.sleep(0.3)

    def push_up(self, steps: int = 1):
        """Do push-ups."""
        if MOCK_MODE:
            logger.info(f"[MOCK] push_up({steps})")
        else:
            self.crawler.do_action("push up", steps, self.speed)
            time.sleep(0.5)

    def wave(self, steps: int = 1):
        """Wave gesture."""
        if MOCK_MODE:
            logger.info(f"[MOCK] wave({steps})")
        else:
            self.crawler.do_action("wave", steps, self.speed)
            time.sleep(0.5)

    def look_left(self, steps: int = 1):
        """Look left."""
        if MOCK_MODE:
            logger.info(f"[MOCK] look_left({steps})")
        else:
            self.crawler.do_action("look left", steps, self.speed)
            time.sleep(0.3)

    def look_right(self, steps: int = 1):
        """Look right."""
        if MOCK_MODE:
            logger.info(f"[MOCK] look_right({steps})")
        else:
            self.crawler.do_action("look right", steps, self.speed)
            time.sleep(0.3)

    def look_up(self, steps: int = 1):
        """Look up."""
        if MOCK_MODE:
            logger.info(f"[MOCK] look_up({steps})")
        else:
            self.crawler.do_action("look up", steps, self.speed)
            time.sleep(0.3)

    def look_down(self, steps: int = 1):
        """Look down."""
        if MOCK_MODE:
            logger.info(f"[MOCK] look_down({steps})")
        else:
            self.crawler.do_action("look down", steps, self.speed)
            time.sleep(0.3)

    def dance(self, steps: int = 1):
        """Dance."""
        if MOCK_MODE:
            logger.info(f"[MOCK] dance({steps})")
        else:
            self.crawler.do_action("dance", steps, self.speed)
            time.sleep(0.5)

    def turn_left_angle(self, steps: int = 1, angle: int = 30):
        """Turn left. angle is the leg swing parameter, actual turn ≈ angle × 0.44 per step."""
        if MOCK_MODE:
            logger.info(f"[MOCK] turn_left_angle({steps}, angle={angle})")
        else:
            self.crawler.angle = angle
            self.crawler.do_action("turn left angle", steps, self.speed)
            time.sleep(0.3)

    def turn_right_angle(self, steps: int = 1, angle: int = 30):
        """Turn right. angle is the leg swing parameter, actual turn ≈ angle × 0.44 per step."""
        if MOCK_MODE:
            logger.info(f"[MOCK] turn_right_angle({steps}, angle={angle})")
        else:
            self.crawler.angle = angle
            self.crawler.do_action("turn right angle", steps, self.speed)
            time.sleep(0.3)

    def nod(self, steps: int = 2):
        """Nod head up and down."""
        if MOCK_MODE:
            logger.info(f"[MOCK] nod({steps})")
        else:
            for _ in range(steps):
                self.crawler.do_action("look up", 1, self.speed)
                time.sleep(0.2)
                self.crawler.do_action("look down", 1, self.speed)
                time.sleep(0.2)
            self.crawler.do_action("stand", 1, self.speed)
            time.sleep(0.3)

    def shake_head(self, steps: int = 2):
        """Shake head side to side."""
        if MOCK_MODE:
            logger.info(f"[MOCK] shake_head({steps})")
        else:
            for _ in range(steps):
                self.crawler.do_action("look left", 1, self.speed)
                time.sleep(0.2)
                self.crawler.do_action("look right", 1, self.speed)
                time.sleep(0.2)
            self.crawler.do_action("stand", 1, self.speed)
            time.sleep(0.3)

    def shake_hand(self, steps: int = 2):
        """Extend front leg for handshake."""
        if MOCK_MODE:
            logger.info(f"[MOCK] shake_hand({steps})")
        else:
            self.crawler.do_action("wave", steps, self.speed)
            time.sleep(0.5)

    def play_dead(self, steps: int = 1):
        """Flip over and play dead."""
        if MOCK_MODE:
            logger.info(f"[MOCK] play_dead({steps})")
        else:
            self.crawler.do_step("sit", self.speed)
            time.sleep(0.5)
        self._standing = False

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
            speaker.announce_obstacle()
        return blocked

    # ── Patrol route ────────────────────────────────────────
    # Each step is (action, args).
    # Modify this list to match your mini city layout.
    PATROL_ROUTE = [
        ("forward", 5),
        ("turn_left", 90),
        ("forward", 2),
        ("turn_right", 90),
        ("forward", 2),
    ]

    def patrol_route(self, should_stop=None):
        """
        Walk a fixed patrol route with obstacle avoidance.

        Executes each step in PATROL_ROUTE sequentially.
        During forward steps, checks ultrasonic before each sub-step.
        If obstacle detected, backs up and waits until clear.

        Args:
            should_stop: callable returning True to abort patrol early.
        """
        if not self._standing:
            self.stand()

        logger.info(f"patrol_route: started, {len(self.PATROL_ROUTE)} steps")

        for i, (action, count) in enumerate(self.PATROL_ROUTE):
            if should_stop and should_stop():
                logger.info(f"patrol_route: aborted at step {i+1}")
                return

            logger.info(f"patrol_route: step {i+1}/{len(self.PATROL_ROUTE)} — {action}({count})")

            if action == "forward":
                self._forward_with_avoidance(count, should_stop)
            elif action == "backward":
                self.backward(count)
            elif action == "turn_left":
                self.turn_left_angle(2, angle=count)
            elif action == "turn_right":
                self.turn_right_angle(2, angle=count)
            else:
                logger.warning(f"patrol_route: unknown action '{action}'")

        logger.info("patrol_route: route completed")

    def _forward_with_avoidance(self, steps, should_stop=None):
        """Walk forward N steps, avoiding obstacles along the way."""
        walked = 0
        while walked < steps:
            if should_stop and should_stop():
                return

            if self.check_obstacle():
                logger.info(f"_forward_with_avoidance: obstacle at step {walked+1}/{steps}, detour")
                self._detour()
            else:
                self.forward(1)
                walked += 1

    def _detour(self):
        """
        Detour around obstacle:
        1. Back up
        2. Turn left 90°
        3. Walk forward (sidestep)
        4. Turn right 90° (face original direction)
        5. Walk forward (pass the obstacle)
        6. Turn right 90°
        7. Walk forward (back to original line)
        8. Turn left 90° (face original direction)
        """
        self.backward(2)
        self.turn_left_angle(2, angle=90)   # ~90° left
        self.forward(1)                      # sidestep
        self.turn_right_angle(2, angle=90)  # face original direction
        self.forward(2)                      # pass obstacle
        self.turn_right_angle(2, angle=90)  # turn back toward original line
        self.forward(1)                      # return to original line
        self.turn_left_angle(2, angle=90)   # face original direction again
