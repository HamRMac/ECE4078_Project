#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Continuous calibration for PenguinPi:
# - Forward: press Enter to START, Enter to STOP, then enter distance [m]
# - Spin    : press Enter to START, Enter to STOP, then enter angle ("720 deg" or "2 rev")
#
# Saves per-robot results under ./param/<robot-id>/ with timestamp + latest files.

import os
import sys
import time
import json
import threading
import datetime as dt
import numpy as np

sys.path.insert(0, "../util")
from pibot import PenguinPi


# ---------- Minimal utils ----------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def derive_robot_id(ppi, fallback_ip: str, cli_robot_id: str | None = None) -> str:
    """Pick a stable per-robot ID: CLI > device-reported > IP-based."""
    if cli_robot_id:
        return cli_robot_id
    for attr in ("get_robot_id", "get_serial", "get_uuid"):
        if hasattr(ppi, attr):
            try:
                rid = getattr(ppi, attr)()
                if rid:
                    return str(rid)
            except Exception:
                pass
    return f"ip-{fallback_ip.replace('.', '-')}" if fallback_ip else "unknown_robot"

def save_calibration(scale: float, baseline: float, base_dir: str, robot_id: str, ip: str, port: int) -> str:
    """Save timestamped calibration files and 'latest' copies, plus metadata JSON."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = ensure_dir(os.path.join(base_dir, robot_id))

    sf_ts = os.path.join(save_dir, f"scale_{ts}.txt")
    bf_ts = os.path.join(save_dir, f"baseline_{ts}.txt")
    np.savetxt(sf_ts, np.array([scale]), delimiter=',')
    np.savetxt(bf_ts, np.array([baseline]), delimiter=',')

    np.savetxt(os.path.join(save_dir, "scale_latest.txt"), np.array([scale]), delimiter=',')
    np.savetxt(os.path.join(save_dir, "baseline_latest.txt"), np.array([baseline]), delimiter=',')

    meta = {
        "robot_id": robot_id,
        "ip": ip,
        "port": port,
        "timestamp": ts,
        "files": {"scale": os.path.basename(sf_ts), "baseline": os.path.basename(bf_ts)},
        "notes": "Scale in m/tick; baseline in meters."
    }
    with open(os.path.join(save_dir, f"calibration_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved calibration:")
    print(f"  Robot ID : {robot_id}")
    print(f"  Dir      : {save_dir}")
    print(f"  Scale    : {scale:.6f} (m/tick)  -> {os.path.basename(sf_ts)} + scale_latest.txt")
    print(f"  Baseline : {baseline:.6f} (m)     -> {os.path.basename(bf_ts)} + baseline_latest.txt")
    return save_dir


# ---------- ENTER-only run helpers ----------
def _wait_for_enter(stop_event: threading.Event, prompt="Press Enter to STOP..."):
    try:
        input(prompt)
    finally:
        stop_event.set()

def run_until_enter(loop_body, poll_hz=20, start_prompt=None, stop_prompt="Press Enter to STOP...") -> float:
    """
    Repeatedly calls loop_body(dt_cmd) until user presses Enter in the terminal.
    Returns elapsed time [s]. Uses only input() (no keyboard libs).
    """
    if start_prompt:
        input(start_prompt)  # Press Enter to START

    stop_event = threading.Event()
    t_listener = threading.Thread(target=_wait_for_enter, args=(stop_event, stop_prompt), daemon=True)
    t_listener.start()

    dt_cmd = 1.0 / float(poll_hz)
    t0 = time.perf_counter()
    try:
        while not stop_event.is_set():
            loop_body(dt_cmd)
            time.sleep(dt_cmd)
    finally:
        # Best-effort stop
        try:
            ppi.set_velocity([0, 0], tick=0, time=0.05)
        except Exception:
            pass
    t1 = time.perf_counter()
    return t1 - t0


# ---------- Parsing helper for spin angle ----------
def parse_rotation_to_rad(s: str) -> float:
    """
    Accepts strings like "720", "720 deg", "1.5 rev", "2 turns".
    Defaults to degrees if units omitted.
    """
    s = s.strip().lower()
    if not s:
        raise ValueError("Empty input.")
    toks = s.split()
    val = float(toks[0])
    unit = toks[1] if len(toks) > 1 else "deg"
    if unit.startswith("deg"):
        return np.deg2rad(val)
    if unit.startswith(("rev", "turn")):
        return val * 2.0 * np.pi
    raise ValueError("Unknown unit. Use 'deg' or 'rev'.")


# ---------- Calibration routines (continuous) ----------
def calibrateWheelRadius_continuous(ppi: PenguinPi, wheel_velocities=(20, 35, 50, 65), poll_hz=20) -> float:
    """
    Forward-drive calibration: for each wheel velocity (ticks/s), runs forward until ENTER is pressed,
    then asks measured distance [m].
      scale_i = distance / (wheel_vel * elapsed)
    Returns average scale [m/tick].
    """
    print("\nFORWARD calibration:")
    print("  1) Press Enter to START each trial")
    print("  2) Press Enter again to STOP")
    print("  3) Enter measured distance in meters\n")

    estimates = []
    for wheel_vel in wheel_velocities:
        def _loop(dt_cmd):
            # Forward: linear command with given wheel tick rate
            ppi.set_velocity([1, 0], tick=wheel_vel, time=dt_cmd)

        elapsed = run_until_enter(
            _loop, poll_hz=poll_hz,
            start_prompt=f"Ready. Trial at {wheel_vel} ticks/s — press Enter to START...",
            stop_prompt="Driving... press Enter to STOP..."
        )
        print(f"Elapsed time: {elapsed:.3f} s")

        while True:
            s = input("Measured distance [m] (e.g., 1.23): ").strip()
            try:
                dist = float(s)
                if dist > 0:
                    break
            except Exception:
                pass
            print("Please enter a positive number.")

        scale_i = dist / (wheel_vel * elapsed)
        print(f"Trial scale estimate: {scale_i:.6f} m/tick\n")
        estimates.append(scale_i)

    scale = float(np.mean(estimates)) if estimates else 0.0
    print(f"Estimated SCALE (avg over {len(estimates)} trials): {scale:.6f} m/tick")
    return scale


def calibrateBaseline_continuous(ppi: PenguinPi, scale: float, turning_ticks=(30, 40, 50), poll_hz=20) -> float:
    """
    Spin-in-place calibration: for each turning_tick (ticks/s), runs until ENTER is pressed,
    then asks measured rotation (deg or rev).
      baseline_i = (2 * scale * turning_tick * elapsed) / measured_angle_rad
    Returns average baseline [m].
    """
    print("\nSPIN calibration (in-place):")
    print("  1) Press Enter to START each trial")
    print("  2) Press Enter again to STOP")
    print("  3) Enter measured rotation as '540 deg' or '1.5 rev'\n")

    estimates = []
    for turning_tick in turning_ticks:
        def _loop(dt_cmd):
            # Pure spin: base tick 0, turning_tick defines in-place rotation
            ppi.set_velocity([0, 1], tick=0, turning_tick=turning_tick, time=dt_cmd)

        elapsed = run_until_enter(
            _loop, poll_hz=poll_hz,
            start_prompt=f"Ready. Trial at turning_tick={turning_tick} ticks/s — press Enter to START...",
            stop_prompt="Spinning... press Enter to STOP..."
        )
        print(f"Elapsed time: {elapsed:.3f} s")

        while True:
            s = input("Measured rotation (e.g., '720 deg' or '1.5 rev'): ").strip()
            try:
                dtheta = parse_rotation_to_rad(s)
                if dtheta > 0:
                    break
            except Exception:
                pass
            print("Please enter like '360 deg' or '1.5 rev'.")

        baseline_i = (2.0 * scale * turning_tick * elapsed) / dtheta
        print(f"Trial baseline estimate: {baseline_i:.6f} m\n")
        estimates.append(baseline_i)

    baseline = float(np.mean(estimates)) if estimates else 0.0
    print(f"Estimated BASELINE (avg over {len(estimates)} trials): {baseline:.6f} m")
    return baseline


# ---------- Main ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PenguinPi continuous calibration (Enter-only).")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--robot-id", metavar='', type=str, default=None,
                        help="Explicit per-robot ID. Defaults to device ID or IP-based fallback.")
    parser.add_argument("--no-per-robot", action="store_true",
                        help="Disable per-robot subfolder; save under ./param/ only.")
    parser.add_argument("--poll-hz", type=int, default=20,
                        help="Command refresh rate while running continuously.")
    parser.add_argument("--fwd-vels", type=str, default="20,35,50,65",
                        help="Comma-separated tick/s for forward trials.")
    parser.add_argument("--spin-vels", type=str, default="30,40,50",
                        help="Comma-separated tick/s for spin trials.")
    args, _ = parser.parse_known_args()

    # Parse velocity lists
    try:
        fwd_vels = tuple(int(v.strip()) for v in args.fwd_vels.split(",") if v.strip())
    except Exception:
        print("Bad --fwd-vels; using defaults.")
        fwd_vels = (20, 35, 50, 65)
    try:
        spin_vels = tuple(int(v.strip()) for v in args.spin_vels.split(",") if v.strip())
    except Exception:
        print("Bad --spin-vels; using defaults.")
        spin_vels = (30, 40, 50)

    ppi = PenguinPi(args.ip, args.port)
    baseDir = ensure_dir(os.path.join(os.getcwd(), "param"))

    try:
        print('Auto Calibrator V1')
        print('Calibrating PiBot scale (continuous forward)...')
        scale = calibrateWheelRadius_continuous(ppi, wheel_velocities=fwd_vels, poll_hz=args.poll_hz)

        print('Calibrating PiBot baseline (continuous spin)...')
        baseline = calibrateBaseline_continuous(ppi, scale, turning_ticks=spin_vels, poll_hz=args.poll_hz)

        if args.no_per_robot:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            np.savetxt(os.path.join(baseDir, f"scale_{ts}.txt"),    np.array([scale]), delimiter=',')
            np.savetxt(os.path.join(baseDir, f"baseline_{ts}.txt"), np.array([baseline]), delimiter=',')
            np.savetxt(os.path.join(baseDir, "scale_latest.txt"),    np.array([scale]), delimiter=',')
            np.savetxt(os.path.join(baseDir, "baseline_latest.txt"), np.array([baseline]), delimiter=',')
            print(f"\nSaved calibration to {baseDir} (shared).")
        else:
            robot_id = derive_robot_id(ppi, args.ip, cli_robot_id=args.robot_id)
            save_calibration(scale, baseline, baseDir, robot_id, args.ip, args.port)

        print('Finished calibration')

    finally:
        # Safety stop
        try:
            ppi.set_velocity([0, 0], tick=0, time=0.05)
        except Exception:
            pass
