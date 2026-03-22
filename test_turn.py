"""Test turn angles for calibration. Run with: python3 -m agent_picrawler.test_turn"""
import time
from picrawler import Picrawler

crawler = Picrawler()
speed = 50

print("Standing up...")
crawler.do_action("stand", 1, speed)
time.sleep(1)

tests = [
    # (label, action, steps, angle, target_degrees)
    # Goal: find how many steps/angle to get exactly 90°
    ("turn_left 5 steps",             "turn left",       5, None, "90°?"),
    ("turn_left_angle 2步 a=90",      "turn left angle", 2, 90,   "90°?"),
    ("turn_left_angle 3步 a=90",      "turn left angle", 3, 90,   "应该比上面多"),
]

for i, (label, action, steps, angle, expect) in enumerate(tests, 1):
    print(f"\n=== Test {i}/{len(tests)}: {label} (预期: {expect}) ===")
    print("对准一个参照物，Press Enter 执行...")
    input()

    crawler.do_action("stand", 1, speed)
    time.sleep(0.5)

    if angle is not None:
        crawler.angle = angle
    crawler.do_action(action, steps, speed)
    time.sleep(0.5)

    print("是90°吗？大于还是小于？")
    result = input("输入结果 (=90 / >90 / <90 / 大概多少度): ")
    print(f"  >> {label} ≈ {result}")

print("\nSitting down...")
crawler.do_step("sit", speed)
print("Done!")
