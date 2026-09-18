import time
import csv
import json
import os
import cv2
import numpy as np
from robomaster import robot, blaster, led

# --------------------------------------------------
# คลาสสำหรับคำนวณ PID Controller
# --------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_limit, self.max_limit = limits
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01

        p_term = self.kp * error
        self.integral += error * dt
        # Anti-windup clamping
        self.integral = max(-50.0, min(50.0, self.integral))
        i_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        output = max(self.min_limit, min(self.max_limit, output))

        self.last_error = error
        self.last_time = now
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


# --------------------------------------------------
# พารามิเตอร์ PID และการควบคุม
# --------------------------------------------------
pid_yaw = PIDController(kp=110.0, ki=0.0, kd=7.0, limits=(-150, 150))
pid_pitch = PIDController(kp=90.0, ki=0.0, kd=5.0, limits=(-100, 100))

# ชนิดการยิง: ยิงเฉพาะกระสุนเจลจริง (WATER_FIRE) เท่านั้น โดยไม่ยิงอินฟราเรดเด็ดขาด
# (สามารถสลับโหมดไม่ยิงอะไรเลย/Aim Only ได้ด้วยปุ่ม [f])
ENABLE_FIRE = True          # True = ยิงกระสุนเจลจริง (WATER_FIRE), False = ไม่ยิงอะไรเลย (Aim & Advance อย่างเดียว)
FIRE_TYPE = blaster.WATER_FIRE  # ปิดการยิงอินฟราเรดเด็ดขาด

# ขนาดพื้นที่ต่ำสุดของวัตถุสี (พิกเซล) เพื่อกรอง Noise เม็ดเล็กๆ
MIN_TARGET_AREA = 700

# --------------------------------------------------
# ตารางช่วงสีในระบบ HSV (โหลดจาก hsv_config.json ที่จูนไว้)
# --------------------------------------------------
CONFIG_FILE = "hsv_config.json"

def load_hsv_configs():
    """โหลดค่า Lower และ Upper จาก hsv_config.json ที่ปรับแต่งไว้จาก hsv_color_tuner.py"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            configs = {}
            for color_name, data in raw_cfg.items():
                ranges = []
                for r in data.get("ranges", []):
                    ranges.append((
                        np.array(r["lower"], dtype=np.uint8),
                        np.array(r["upper"], dtype=np.uint8)
                    ))
                configs[color_name] = {
                    "ranges": ranges,
                    "draw_color": tuple(data.get("draw_color", [0, 255, 0])),
                    "led_rgb": tuple(data.get("led_rgb", [255, 255, 255])),
                    "name_th": data.get("name_th", color_name)
                }
            print(f">> [CONFIG] โหลดค่า Lower/Upper HSV จาก '{CONFIG_FILE}' สำเร็จ")
            return configs
        except Exception as e:
            print(f"[!] ไม่สามารถอ่าน {CONFIG_FILE}: {e}")

    # Fallback ค่าเริ่มต้นหากไม่พบไฟล์
    return {
        "Red": {
            "ranges": [
                (np.array([0, 100, 70]), np.array([10, 255, 255])),
                (np.array([165, 100, 70]), np.array([180, 255, 255]))
            ],
            "draw_color": (0, 0, 255),
            "led_rgb": (255, 0, 0),
            "name_th": "แดง"
        },
        "Green": {
            "ranges": [
                (np.array([35, 80, 70]), np.array([85, 255, 255]))
            ],
            "draw_color": (0, 255, 0),
            "led_rgb": (0, 255, 0),
            "name_th": "เขียว"
        },
        "Blue": {
            "ranges": [
                (np.array([95, 100, 70]), np.array([130, 255, 255]))
            ],
            "draw_color": (255, 130, 0),
            "led_rgb": (0, 100, 255),
            "name_th": "น้ำเงิน"
        },
        "Yellow": {
            "ranges": [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ],
            "draw_color": (0, 220, 255),
            "led_rgb": (255, 255, 0),
            "name_th": "เหลือง"
        }
    }

COLOR_CONFIGS = load_hsv_configs()

# โหมดการทำงาน: "LEARNING" (เรียนรู้ลำดับ) -> "TRACKING" (เล็งและยิงตามลำดับ)
app_mode = "LEARNING"

# ลำดับเป้าหมายที่เรียนรู้ได้จากซ้ายไปขวา
learned_sequence = []
current_step = 0

# สถานะการทำงาน
is_shooting = False
is_firing_now = 0
last_active_label = "None"

# --------------------------------------------------
# บันทึกข้อมูล Time Response (CSV)
# --------------------------------------------------
data_log = []
start_record_time = 0.0
current_pitch_angle = 0.0
current_yaw_angle = 0.0


def on_gimbal_angle(angle_info):
    """Callback บันทึกมุม Pitch, Yaw จาก Gimbal"""
    global current_pitch_angle, current_yaw_angle, data_log, start_record_time
    pitch_angle, yaw_angle, pitch_ground, yaw_ground = angle_info
    current_pitch_angle = pitch_angle
    current_yaw_angle = yaw_angle

    if start_record_time > 0 and app_mode == "TRACKING":
        elapsed = time.time() - start_record_time
        data_log.append([
            round(elapsed, 4),
            round(pitch_angle, 2),
            round(yaw_angle, 2),
            last_active_label,
            is_firing_now
        ])


def classify_shape(cnt):
    """
    ตรวจสอบรูปทรง: รับเฉพาะ 'Circle' (ทรงกลม) หรือ 'Rectangle' (สี่เหลี่ยม)
    คืนค่า (shape_name, approx_poly) หากไม่ใช่ทั้งคู่คืนค่า (None, None)
    """
    area = cv2.contourArea(cnt)
    if area < MIN_TARGET_AREA:
        return None, None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None, None

    # 1. ตรวจสอบความเป็นทรงกลม (Circularity & Minimum Enclosing Circle)
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    _, radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * (radius ** 2)
    circle_ratio = area / circle_area if circle_area > 0 else 0

    # 2. ตรวจสอบความเป็นสี่เหลี่ยม (Polygon Approximation & Extent)
    approx = cv2.approxPolyDP(cnt, 0.038 * perimeter, True)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    bbox_area = bw * bh
    extent = area / bbox_area if bbox_area > 0 else 0

    is_4_corners = (len(approx) == 4) and cv2.isContourConvex(approx)

    # เงื่อนไขทรงกลม
    if circularity >= 0.70 and circle_ratio >= 0.68:
        return "Circle", approx

    # เงื่อนไขสี่เหลี่ยม
    if (is_4_corners and extent >= 0.65) or (extent >= 0.76 and len(approx) in (4, 5)):
        return "Rectangle", approx

    return None, None


def detect_all_targets(img):
    """
    ค้นหาเฉพาะวัตถุรูป 'ทรงกลม' (Circle) และ 'สี่เหลี่ยม' (Rectangle)
    จากทุกสีที่กำหนด และเรียงลำดับเป้าหมายจาก "ซ้ายไปขวา" ตามพิกัด X
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = img.shape
    targets = []

    for color_name, color_cfg in COLOR_CONFIGS.items():
        combined_mask = None
        for lower, upper in color_cfg["ranges"]:
            mask_part = cv2.inRange(hsv, lower, upper)
            if combined_mask is None:
                combined_mask = mask_part
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask_part)

        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            shape_type, approx = classify_shape(cnt)
            if shape_type is None:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bx, by, bw, bh = cv2.boundingRect(cnt)

            targets.append({
                "color": color_name,
                "shape": shape_type,
                "label": f"{color_name} {shape_type}",
                "name_th": f"{'กลม' if shape_type == 'Circle' else 'สี่เหลี่ยม'}{color_cfg['name_th']}",
                "center": (cx, cy),
                "norm_x": cx / w,
                "norm_y": cy / h,
                "bbox": (bx, by, bw, bh),
                "contour": cnt,
                "approx": approx,
                "draw_color": color_cfg["draw_color"],
                "led_rgb": color_cfg["led_rgb"],
                "area": cv2.contourArea(cnt)
            })

    # เรียงลำดับเป้าหมายทั้งหมดจาก "ซ้ายไปขวา" ตามพิกัด x
    targets.sort(key=lambda t: t["center"][0])
    return targets


def fire_single_shot(ep_blaster, ep_led):
    """ยิงกระสุน 1 นัด (กระสุนเจลจริง WATER_FIRE เท่านั้น โดยไม่ยิงอินฟราเรดเด็ดขาด)"""
    global is_firing_now
    if not ENABLE_FIRE:
        print("\n>> [FIRE OFF] โหมดไม่ยิงกระสุน (ขยับเป้าหมายอย่างเดียว ไม่มีการยิงใดๆ)")
        return

    is_firing_now = 1
    print("\n💥 >> [FIRE!] ยิงกระสุนเจลจริง (WATER_FIRE) 1 นัด (ไม่ยิงอินฟราเรด) << 💥")

    ep_led.set_led(comp=led.COMP_TOP_ALL, r=255, g=0, b=0, effect=led.EFFECT_ON)
    ep_blaster.fire(fire_type=blaster.WATER_FIRE, times=1)
    time.sleep(0.3)
    ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)

    is_firing_now = 0
    pid_yaw.reset()
    pid_pitch.reset()


def shoot_and_advance_step(ep_blaster, ep_led):
    """ยิง 1 นัด แล้วเลื่อนไปยังเป้าหมายถัดไปในลำดับที่เรียนรู้ไว้"""
    global current_step, is_shooting

    is_shooting = True
    fire_single_shot(ep_blaster, ep_led)

    if len(learned_sequence) > 0:
        current_step = (current_step + 1) % len(learned_sequence)
        next_tgt = learned_sequence[current_step]
        print(f">> [ADVANCE] เลื่อนไปเป้าหมายถัดไปตามลำดับ: ขั้นที่ {current_step+1}/{len(learned_sequence)} -> [{next_tgt['color']} {next_tgt['shape']}]")

    time.sleep(0.3)
    is_shooting = False


def main():
    global start_record_time, app_mode, learned_sequence, current_step
    global is_shooting, last_active_label, ENABLE_FIRE, COLOR_CONFIGS

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_led = ep_robot.led
    ep_camera = ep_robot.camera

    # ดับไฟเริ่มต้น
    ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
    ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)

    print("เปิดกล้องและตั้งศูนย์ Gimbal...")
    ep_camera.start_video_stream(display=False)
    ep_gimbal.recenter().wait_for_completed()
    time.sleep(1)

    # รับข้อมูลมุมกิมบอลที่ 20Hz
    ep_gimbal.sub_angle(freq=20, callback=on_gimbal_angle)
    start_record_time = time.time()

    print("\n" + "="*75)
    print("  ระบบเรียนรู้สีและรูปทรงเป้าหมาย (ซ้ายไปขวา) -> ล็อกลำดับยิงอัตโนมัติ")
    print("="*75)
    print("  ขั้นตอนการทำงาน:")
    print("   1. [LEARNING MODE] เปิดขึ้นมา ระบบจะสแกนหา 'กลม / สี่เหลี่ยม' และสีทั้งหมด")
    print("      เรียงลำดับจาก ซ้าย -> ขวา ให้เห็นบนหน้าจอ")
    print("      >> กด [SPACEBAR] หรือ [ENTER] เพื่อบันทึกลำดับนี้และเริ่มเข้าโหมดเล็งยิง")
    print("   2. [TRACKING MODE] ระบบจะเล็งเป้าหมายตามลำดับที่เรียนรู้ไว้ทีละเป้า:")
    print("      - [SPACEBAR] / [y] : ยิง 1 นัด และขยับไปเป้าถัดไปในลำดับ")
    print("      - [n]              : สั่งข้ามไปเป้าถัดไป")
    print("      - [p]              : ย้อนกลับไปเป้าก่อนหน้า")
    print("      - [l]              : กลับไปโหมดเรียนรู้สแกนลำดับใหม่ (Re-Learn)")
    print("      - [r]              : รีเซ็ตศูนย์กิมบอล")
    print("      - [q]              : ออกจากโปรแกรม")
    print("="*75 + "\n")

    is_locked = False
    cooldown_until = 0.0

    while True:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            continue

        h, w, _ = img.shape
        center_screen_x = w // 2
        center_screen_y = h // 2

        # วาดเส้น Crosshair กึ่งกลางกล้อง
        cv2.line(img, (center_screen_x - 25, center_screen_y), (center_screen_x + 25, center_screen_y), (255, 255, 255), 1)
        cv2.line(img, (center_screen_x, center_screen_y - 25), (center_screen_x, center_screen_y + 25), (255, 255, 255), 1)

        # ค้นหาเป้าหมายทั้งหมดในเฟรมปัจจุบัน (เฉพาะกลมและสี่เหลี่ยม เรียงซ้ายไปขวา)
        live_targets = detect_all_targets(img)

        # ==================================================
        # 1. โหมดเรียนรู้ลำดับเป้าหมาย (LEARNING MODE)
        # ==================================================
        if app_mode == "LEARNING":
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

            # ไฟกะพริบสีฟ้าแสดงสถานะกำลังสแกนเรียนรู้
            ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=150, b=255, effect=led.EFFECT_BREATH)

            # วาดเป้าหมายทุกตัวที่ตรวจพบ พร้อมลำดับจากซ้ายไปขวา
            for idx, tgt in enumerate(live_targets):
                bx, by, bw, bh = tgt["bbox"]
                cx, cy = tgt["center"]
                draw_color = tgt["draw_color"]

                if tgt["shape"] == "Circle":
                    radius = max(bw, bh) // 2
                    cv2.circle(img, (cx, cy), radius, draw_color, 2)
                else:
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), draw_color, 2)

                # ป้ายลำดับ 1, 2, 3...
                tag = f"#{idx+1} {tgt['color']} {tgt['shape']}"
                cv2.rectangle(img, (bx, max(15, by - 22)), (bx + len(tag) * 9 + 8, max(18, by)), (0, 0, 0), -1)
                cv2.putText(img, tag, (bx + 3, max(15, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            # UI ส่วนหัวแจ้งสถานะ
            cv2.putText(img, "=== [1. LEARNING / SCAN MODE] ===", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            seq_text = "Detected (L->R): " + (" -> ".join([f"#{i+1}:{t['color']}_{t['shape'][:4]}" for i, t in enumerate(live_targets)]) if live_targets else "Waiting for targets...")
            cv2.putText(img, seq_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # แถบคำสั่งยืนยันด้านล่าง
            overlay = img.copy()
            cv2.rectangle(overlay, (10, h - 75), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

            if len(live_targets) > 0:
                prompt_str = f"Found {len(live_targets)} targets! Press [SPACEBAR] to LOCK THIS SEQUENCE & START"
                cv2.putText(img, prompt_str, (25, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Searching for Circle / Rectangle targets... Please place targets in view.", (25, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

            cv2.imshow("RoboMaster Target Sequence Controller", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("ผู้ใช้สั่งหยุดโปรแกรม")
                break
            elif key == ord('r'):
                ep_gimbal.recenter().wait_for_completed()
            elif key in (ord('c'), ord('C')):
                COLOR_CONFIGS = load_hsv_configs()
                print(">> [RELOAD] โหลดค่า Lower/Upper HSV ใหม่จาก hsv_config.json สำเร็จ!")
            elif key in (ord(' '), 13):  # SPACE หรือ ENTER เพื่อบันทึกลำดับ
                if len(live_targets) > 0:
                    learned_sequence = [
                        {
                            "color": t["color"],
                            "shape": t["shape"],
                            "label": t["label"],
                            "draw_color": t["draw_color"],
                            "led_rgb": t["led_rgb"],
                            "name_th": t["name_th"]
                        }
                        for t in live_targets
                    ]
                    current_step = 0
                    app_mode = "TRACKING"
                    ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=255, b=0, effect=led.EFFECT_ON)
                    time.sleep(0.4)
                    ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)

                    print("\n" + "="*60)
                    print(f"🎯 [SEQUENCE LOCKED!] บันทึกลำดับเป้าหมาย {len(learned_sequence)} เป้า เรียบร้อยแล้ว:")
                    for i, t in enumerate(learned_sequence):
                        print(f"   เป้า #{i+1} : {t['color']} {t['shape']} ({t['name_th']})")
                    print("เริ่มเข้าสู่โหมดเล็งและยิงตามลำดับ!")
                    print("="*60 + "\n")
                else:
                    print(">> ยังไม่พบเป้าหมายที่ถูกต้อง กรุณาวางเป้าให้อยู่ในมุมกล้อง")

            continue

        # ==================================================
        # 2. โหมดเล็งและยิงตามลำดับที่เรียนรู้ (TRACKING MODE)
        # ==================================================
        active_step_tgt = learned_sequence[current_step]
        last_active_label = f"Step{current_step+1}_{active_step_tgt['color']}_{active_step_tgt['shape']}"

        # ค้นหาเป้าหมายในภาพที่ตรงกับเงื่อนไขของ Step ปัจจุบัน (สี + รูปทรง)
        matched_targets = [
            t for t in live_targets
            if t["color"] == active_step_tgt["color"] and t["shape"] == active_step_tgt["shape"]
        ]

        target_to_aim = None
        if matched_targets:
            # ถ้ามีหลายตัวที่ตรงกัน ให้เลือกตัวที่มีตำแหน่งใกล้เคียงลำดับเดิม หรือตัวที่ขนาดใหญ่สุด
            matched_targets.sort(key=lambda t: t["area"], reverse=True)
            target_to_aim = matched_targets[0]

        # วาดเป้าหมายทั้งหมดและเน้นเป้าหมายของ Step ปัจจุบัน
        for t in live_targets:
            bx, by, bw, bh = t["bbox"]
            is_current = (target_to_aim is not None and t is target_to_aim)
            box_col = (0, 255, 255) if is_current else (100, 100, 100)

            if t["shape"] == "Circle":
                cv2.circle(img, t["center"], max(bw, bh) // 2, box_col, 2 if is_current else 1)
            else:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), box_col, 2 if is_current else 1)

            cv2.putText(img, f"{t['color']} {t['shape']}", (bx, max(15, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1)

        in_cooldown = time.time() < cooldown_until

        if target_to_aim is not None and not in_cooldown and not is_shooting:
            norm_x = target_to_aim["norm_x"]
            norm_y = target_to_aim["norm_y"]
            cx, cy = target_to_aim["center"]

            # วาดเส้นเล็งและวงกลมล็อกเป้า
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), 14, (0, 255, 255), 2)
            cv2.line(img, (center_screen_x, center_screen_y), (cx, cy), (0, 255, 255), 1)

            err_x = norm_x - 0.5
            err_y = 0.5 - norm_y

            # ตรวจสอบความแม่นยำกึ่งกลางจอ (Error < 3%)
            if abs(err_x) < 0.01 and abs(err_y) < 0.01:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

                if not is_locked:
                    is_locked = True
                    r_c, g_c, b_c = active_step_tgt["led_rgb"]
                    ep_led.set_led(comp=led.COMP_TOP_ALL, r=r_c, g=g_c, b=b_c, effect=led.EFFECT_ON)
                    fire_msg = "ยิงกระสุนเจล (WATER_FIRE)" if ENABLE_FIRE else "ไม่ยิง (Aim Only)"
                    print(f"\n[🎯 LOCKED!] ล็อกเป้าหมายขั้นที่ #{current_step+1} [{active_step_tgt['color']} {active_step_tgt['shape']}] เข้ากึ่งกลางแล้ว!")
                    print(f"--> โหมดปัจจุบัน: {fire_msg} | กด [SPACEBAR] เพื่อดำเนินการและไปยังเป้าถัดไป")
            else:
                if is_locked:
                    ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
                is_locked = False

                yaw_speed = pid_yaw.compute(err_x)
                pitch_speed = pid_pitch.compute(err_y)
                ep_gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)
        else:
            if is_locked:
                ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
                ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
            is_locked = False
            if not is_shooting:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        # --------------------------------------------------
        # UI แสดงผลโหมด TRACKING
        # --------------------------------------------------
        header_text = f"Step ({current_step+1}/{len(learned_sequence)}): [{active_step_tgt['color']} {active_step_tgt['shape']}] | Yaw: {current_yaw_angle:.1f} Pitch: {current_pitch_angle:.1f}"
        cv2.putText(img, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # แถบแสดงลำดับทั้งหมดที่ล็อกไว้
        seq_bar = "Sequence: " + " -> ".join([
            f"[#{i+1} {s['color']} {s['shape'][:4]}]" if i == current_step
            else f"#{i+1} {s['color']} {s['shape'][:4]}"
            for i, s in enumerate(learned_sequence)
        ])
        cv2.putText(img, seq_bar, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 100), 1)

        # ป้ายสถานะด้านล่าง
        fire_status_str = "WATER_FIRE (เจล)" if ENABLE_FIRE else "NO FIRE (ขยับอย่างเดียว)"
        if is_locked and target_to_aim is not None:
            overlay = img.copy()
            cv2.rectangle(overlay, (10, h - 85), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

            next_name = learned_sequence[(current_step + 1) % len(learned_sequence)]["label"]
            msg = f">> LOCKED STEP {current_step+1}! [{fire_status_str}] [SPACE]: GO & NEXT ({next_name}) | [F]: TOGGLE FIRE | [L]: RE-LEARN <<"
            cv2.putText(img, msg, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 2)
        elif target_to_aim is not None:
            cv2.putText(img, f"Aiming at Step {current_step+1}: {active_step_tgt['color']} {active_step_tgt['shape']}... [{fire_status_str}]", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(img, f"Searching for Step {current_step+1}: {active_step_tgt['color']} {active_step_tgt['shape']}... [{fire_status_str}] [N] Skip | [L] Re-learn", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 2)

        cv2.imshow("RoboMaster Target Sequence Controller", img)

        # --------------------------------------------------
        # ปุ่มควบคุมโหมด TRACKING
        # --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("ผู้ใช้กด 'q' เพื่อหยุดการทำงาน")
            break
        elif key in (ord('f'), ord('F')):
            ENABLE_FIRE = not ENABLE_FIRE
            status_desc = "ยิงกระสุนเจลจริง (WATER_FIRE - ไม่ยิงอินฟราเรด)" if ENABLE_FIRE else "ปิดการยิง (Aim & Move Only - ไม่ยิงอะไรเลย)"
            print(f">> [TOGGLE] สลับโหมดการยิงเป็น: {status_desc}")
        elif key == ord('r'):
            ep_gimbal.recenter().wait_for_completed()
            pid_yaw.reset()
            pid_pitch.reset()
        elif key in (ord('c'), ord('C')):
            COLOR_CONFIGS = load_hsv_configs()
            print(">> [RELOAD] โหลดค่า Lower/Upper HSV ใหม่จาก hsv_config.json สำเร็จ!")
        elif key in (ord('l'), ord('L')):
            # สั่งกลับไปโหมดเรียนรู้สแกนลำดับใหม่
            print("\n>> [RE-LEARN] สั่งกลับเข้าสู่โหมดสแกนเรียนรู้ลำดับใหม่...")
            app_mode = "LEARNING"
            is_locked = False
            ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
            ep_led.set_led(comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
            pid_yaw.reset()
            pid_pitch.reset()
            continue
        elif key in (ord('n'), ord('N'), ord('s'), ord('S'), 9):  # ข้ามไปเป้าถัดไป
            current_step = (current_step + 1) % len(learned_sequence)
            next_tgt = learned_sequence[current_step]
            print(f">> [NEXT] สั่งข้ามไปยังขั้นที่ #{current_step+1} [{next_tgt['color']} {next_tgt['shape']}]")
            is_locked = False
            pid_yaw.reset()
            pid_pitch.reset()
            cooldown_until = time.time() + 0.4
        elif key in (ord('p'), ord('P')):  # ย้อนกลับเป้าก่อนหน้า
            current_step = (current_step - 1) % len(learned_sequence)
            prev_tgt = learned_sequence[current_step]
            print(f">> [PREV] สั่งย้อนกลับไปยังขั้นที่ #{current_step+1} [{prev_tgt['color']} {prev_tgt['shape']}]")
            is_locked = False
            pid_yaw.reset()
            pid_pitch.reset()
            cooldown_until = time.time() + 0.4
        elif ord('1') <= key <= ord('9'):  # เลือกขั้นโดยตรง
            sel_idx = key - ord('1')
            if sel_idx < len(learned_sequence):
                current_step = sel_idx
                st = learned_sequence[current_step]
                print(f">> [SELECT] สั่งเลือกขั้นที่ #{current_step+1} [{st['color']} {st['shape']}] โดยตรง")
                is_locked = False
                pid_yaw.reset()
                pid_pitch.reset()
                cooldown_until = time.time() + 0.4
        elif key in (ord(' '), 13, ord('y'), ord('Y')):  # SPACE / ENTER / Y
            if is_locked and target_to_aim is not None:
                shoot_and_advance_step(ep_blaster, ep_led)
                is_locked = False
                cooldown_until = time.time() + 1.2
            else:
                fire_single_shot(ep_blaster, ep_led)
                cooldown_until = time.time() + 1.0

    # --------------------------------------------------
    # ปิดการทำงานและบันทึกข้อมูล
    # --------------------------------------------------
    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    ep_gimbal.unsub_angle()
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
    ep_led.set_led(comp=led.COMP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
    ep_robot.close()

    csv_filename = "sequence_learned_response.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "pitch_angle", "yaw_angle", "target_step", "is_fired"])
        writer.writerows(data_log)

    print(f"\n>> บันทึกข้อมูล Time Response เรียบร้อยแล้ว: {csv_filename} (จำนวน {len(data_log)} บรรทัด)")


if __name__ == '__main__':
    main()
