import os
import sys
import csv
import numpy as np

# ป้องกันปัญหา UnicodeEncodeError บน Windows terminal (cp874/cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# กำหนดสไตล์กราฟให้ดูทันสมัย สะอาด สวยงาม
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False


def find_csv_file():
    """ค้นหาไฟล์ CSV ข้อมูลการยิง"""
    candidates = [
        "sequence_auto_infrared_response.csv",
        os.path.join("data", "sequence_auto_infrared_response.csv"),
        "sequence_learned_response.csv",
        os.path.join("data", "sequence_learned_response.csv"),
    ]
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def parse_shooting_data(csv_path):
    """อ่านข้อมูลจาก CSV และจัดกลุ่ม Firing Bursts"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        raise ValueError(f"ไฟล์ {csv_path} ไม่มีข้อมูล")

    raw_times = np.array([float(r["time_sec"]) for r in reader])
    t0 = raw_times[0]
    time_sec = raw_times - t0  # เริ่มนับจากวินาทีที่ 0.0

    pitch_angles = np.array([float(r["pitch_angle"]) for r in reader])
    yaw_angles = np.array([float(r["yaw_angle"]) for r in reader])
    is_fired = np.array([int(r["is_fired"]) for r in reader])
    target_steps = [r["target_step"] for r in reader]

    # รวมกลุ่มการยิง (Grouping Firing Bursts)
    bursts = []
    current_burst = []
    for i, fired in enumerate(is_fired):
        if fired == 1:
            current_burst.append(i)
        elif current_burst:
            bursts.append(current_burst)
            current_burst = []
    if current_burst:
        bursts.append(current_burst)

    burst_info = []
    for b in bursts:
        start_idx = b[0]
        end_idx = b[-1]
        mid_idx = b[len(b) // 2]
        burst_info.append({
            "start_time": time_sec[start_idx],
            "end_time": time_sec[end_idx],
            "mid_time": time_sec[mid_idx],
            "yaw": yaw_angles[mid_idx],
            "pitch": pitch_angles[mid_idx],
            "target": target_steps[mid_idx],
            "duration": time_sec[end_idx] - time_sec[start_idx],
            "points": len(b)
        })

    return {
        "time_sec": time_sec,
        "pitch_angles": pitch_angles,
        "yaw_angles": yaw_angles,
        "is_fired": is_fired,
        "target_steps": target_steps,
        "bursts": burst_info,
        "t0": t0
    }


def plot_shooting_response(data, output_img_path="plots/shooting_response_plot.png"):
    """สร้างกราฟวิเคราะห์มุมกิมบอลและการยิงอินฟราเรด"""
    os.makedirs(os.path.dirname(output_img_path) or ".", exist_ok=True)

    t = data["time_sec"]
    yaw = data["yaw_angles"]
    pitch = data["pitch_angles"]
    fired = data["is_fired"]
    targets = data["target_steps"]
    bursts = data["bursts"]

    # กำหนดสีประจำเป้าหมาย
    step_colors = {
        "Step1_Green": "#2ecc71",
        "Step2_Blue": "#3498db",
        "Step3_Red": "#e74c3c",
        "Step4_Yellow": "#f1c40f",
        "None": "#95a5a6"
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})

    # -------------------------------------------------------------
    # กราฟบน (AX1): มุม Yaw และ Pitch ของ Gimbal + ไฮไลต์เป้าหมาย
    # -------------------------------------------------------------
    ax1.plot(t, yaw, label="Yaw Angle (°)", color="#1f77b4", linewidth=2.2, zorder=4)
    ax1.plot(t, pitch, label="Pitch Angle (°)", color="#9b59b6", linewidth=1.8, linestyle="--", zorder=4)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # คำนวณช่วงขอบเขตแกน Y เพื่อไม่ให้ข้อความหรือป้ายชนขอบ
    y_min = min(yaw.min(), pitch.min(), -35) - 22
    y_max = max(yaw.max(), 32) + 26
    ax1.set_ylim(y_min, y_max)

    # วาดแถบสีพื้นหลังตามแต่ละ Step ของเป้าหมาย
    current_tgt = None
    start_t = t[0]
    for i in range(len(targets)):
        tgt = targets[i]
        if tgt != current_tgt or i == len(targets) - 1:
            if current_tgt is not None:
                bg_col = "#ecf0f1"
                for k, col in step_colors.items():
                    if k in current_tgt:
                        bg_col = col
                        break
                ax1.axvspan(start_t, t[i], alpha=0.12, color=bg_col, zorder=1)
                mid_t = (start_t + t[i]) / 2
                clean_name = current_tgt.replace("Step", "Step ").replace("_", " ")
                ax1.text(mid_t, y_max - 6, clean_name,
                         ha="center", va="top", fontsize=9.5, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.28", facecolor=bg_col, alpha=0.45, edgecolor="none"),
                         zorder=5)
            current_tgt = tgt
            start_t = t[i]

    # ไฮไลต์จุดยิง (Burst Highlights & Markers)
    for idx, b in enumerate(bursts):
        ax1.axvspan(b["start_time"], b["end_time"], color="#e74c3c", alpha=0.35, zorder=3)
        ax1.scatter([b["mid_time"]], [b["yaw"]], color="#d35400", edgecolors="red", s=130, marker="*", zorder=6)
        
        # วางกล่องข้อความ Shot สลับบน/ล่างอย่างสวยงาม
        offset_y = 16 if b["yaw"] >= 0 else -18
        ax1.annotate(
            f"[FIRE] Shot #{idx+1} (IR)\nYaw: {b['yaw']:.1f} deg\nt = {b['mid_time']:.1f}s",
            xy=(b["mid_time"], b["yaw"]),
            xytext=(b["mid_time"], b["yaw"] + offset_y),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#900C3F",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
            zorder=7
        )

    ax1.set_ylabel("Gimbal Angle (Degrees)", fontsize=11, fontweight="bold")
    ax1.set_title("RoboMaster Auto-Infrared Sequence Response Analysis\nPID Target Tracking & Automatic Infrared Firing Response",
                  fontsize=13, fontweight="bold", pad=14)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#dcdde1")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # กราฟล่าง (AX2): สถานะการยิงอินฟราเรด (is_fired pulse)
    # -------------------------------------------------------------
    ax2.step(t, fired, where="post", color="#c0392b", linewidth=2, label="IR Fire Signal (1=Fired, 0=Aiming)", zorder=4)
    ax2.fill_between(t, 0, fired, step="post", alpha=0.35, color="#e74c3c", zorder=3)

    for idx, b in enumerate(bursts):
        ax2.axvspan(b["start_time"], b["end_time"], color="#e74c3c", alpha=0.5, zorder=2)
        ax2.text(b["mid_time"], 1.15, f"Shot #{idx+1}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#900C3F")

    ax2.set_ylabel("Fire State", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time Elapsed (seconds)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.15, 1.4)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["0 (Ready)", "1 (IR FIRE)"])
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    print(f">> [GRAPH] บันทึกรูปกราฟเรียบร้อยแล้ว: {output_img_path}")

    # สร้างรูป copy ใน root ด้วยเพื่อความสะดวก
    root_copy = os.path.basename(output_img_path)
    if root_copy != output_img_path:
        plt.savefig(root_copy, dpi=300, bbox_inches="tight")
        print(f">> [GRAPH] บันทึกสำเนาไว้ที่ root: {root_copy}")

    return output_img_path


def print_summary(data):
    """แสดงข้อมูลสถิติสรุปการทดสอบ"""
    t = data["time_sec"]
    bursts = data["bursts"]
    print("\n" + "="*60)
    print(" [*] สรุปผลการทดสอบระบบยิงอินฟราเรดอัตโนมัติ (Auto-IR Summary)")
    print("="*60)
    print(f" - ระยะเวลาทดสอบทั้งหมด: {t[-1] - t[0]:.2f} วินาที")
    print(f" - จำนวนการยิงที่สำเร็จ  : {len(bursts)} ครั้ง")
    for i, b in enumerate(bursts):
        print(f"   [{i+1}] เวลา: {b['mid_time']:.2f}s | เป้าหมาย: {b['target']} | มุม Yaw: {b['yaw']:+.1f} deg | Pitch: {b['pitch']:+.1f} deg")
    print("="*60 + "\n")


def main():
    csv_path = find_csv_file()
    if not csv_path:
        print("[!] ไม่พบไฟล์ CSV ข้อมูลการยิง กรุณาระบุ path หรือรันโปรแกรมยิงเพื่อสร้างข้อมูล")
        return

    print(f">> โหลดข้อมูลจาก: {csv_path}")
    data = parse_shooting_data(csv_path)
    print_summary(data)
    out_path = plot_shooting_response(data, output_img_path="plots/shooting_response_plot.png")

    if "--show" in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
