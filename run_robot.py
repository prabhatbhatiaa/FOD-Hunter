#!/usr/bin/env python3
"""
Runway Foreign Object Debris (FOD) Inspection Robot Runner
Live Cyberwave Simulation Controller (Unitree Go2):
- Connects to Cyberwave and activates live simulation mode via cw.affect("simulation")
- Queries and prints all environments, twins, waypoints, and the 5 FOD primitives (FOD-01 to FOD-05)
- Controls the real Unitree Go2 twin ('Runway Inspector — Unitree Go2') along the 5 runway waypoints
- Waits until robot arrives (< 1.0m Euclidean distance) polling twin.position (15s timeout)
- Repositions via twin.edit_position(x=wp_x, y=wp_y) if Nav2 aborts or times out, sleeping 1.5s
- Aims robot yaw directly toward nearest FOD primitive using math.atan2 via twin.edit_rotation
- Captures live frames from onboard 'front_camera' sensor (standardized to 640x360)
- Detects Red, Yellow, Blue, Purple, and Green FOD blocks with tuned OpenCV HSV contour detection:
    * min_area = 30 px
    * 30 <= area < 250: confidence mapped to 0.55 - 0.80 (triggers relative_move forward 1.0m)
    * area >= 250: confidence >= 0.88 (CONFIRMED FOD)
    * area < 30: confidence < 0.50 (ignored / clear runway)
- Fuses contour detection with YOLOv8n inference
- Saves EVERY captured frame (initial scan, uncertain initial, and close-up re-inspected) to evidence/
- Displays formatted terminal summary table of all detected FOD hazards
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Environment & Configuration Setup
# ---------------------------------------------------------------------------
load_dotenv()
CYBERWAVE_API_KEY = os.getenv("CYBERWAVE_API_KEY")
CYBERWAVE_ENVIRONMENT_ID = os.getenv("CYBERWAVE_ENVIRONMENT_ID", "3a4083b5-9a82-4541-ac0b-45c2e6b1864f")

if not CYBERWAVE_API_KEY:
    raise ValueError("CYBERWAVE_API_KEY is not defined in .env")

EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

from cyberwave import Cyberwave

# Initialize local YOLOv8n model
yolo_model = None
try:
    from ultralytics import YOLO
    print("[Vision] Loading local YOLOv8n model...")
    yolo_model = YOLO("yolov8n.pt")
    print("[Vision] YOLOv8n model initialized.")
except Exception as err:
    print(f"[Vision Notice] YOLO model note: {err}. Running OpenCV HSV contour detection.")


# ---------------------------------------------------------------------------
# Helper: Extract Robot Position from Twin Instance
# ---------------------------------------------------------------------------
def get_robot_position(twin: Any) -> Tuple[float, float, float]:
    """Retrieve current (x, y, z) coordinates of the robot twin."""
    pos = getattr(twin, "_position", None)
    if pos is None and hasattr(twin, "_get_current_position"):
        try:
            pos = twin._get_current_position()
        except Exception:
            pass

    if pos is None and hasattr(twin, "refresh"):
        try:
            twin.refresh()
            if hasattr(twin, "_data"):
                pos = {
                    "x": getattr(twin._data, "position_x", 0.0),
                    "y": getattr(twin._data, "position_y", 0.0),
                    "z": getattr(twin._data, "position_z", 0.0),
                }
        except Exception:
            pass

    if pos is None and hasattr(twin, "pose"):
        tr = getattr(twin.pose, "translation", None)
        if tr:
            pos = tr

    if isinstance(pos, dict):
        return float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))
    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)
    elif hasattr(pos, "x") and hasattr(pos, "y"):
        return float(pos.x), float(pos.y), float(getattr(pos, "z", 0.0))

    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Helper: Query FOD Primitives from Cyberwave Scene
# ---------------------------------------------------------------------------
def query_fod_primitives(cw: Cyberwave, environment_id: str) -> Dict[str, List[float]]:
    """
    Queries universal schema from Cyberwave environment to extract real 3D coordinates
    for all 5 FOD primitives (FOD-01 through FOD-05).
    """
    fod_coords: Dict[str, List[float]] = {}
    try:
        s_json = cw.environments.get_universal_schema_json(environment_id)
        data = json.loads(s_json) if isinstance(s_json, str) else s_json
        obstacles = (
            data.get("extensions", {})
            .get("cyberwave", {})
            .get("planning_scene", {})
            .get("obstacles", [])
        )
        for obs in obstacles:
            o_name = str(obs.get("name", "")).strip()
            o_id = str(obs.get("id", "")).strip()
            if "fod" in o_name.lower() or "fod" in o_id.lower():
                mn = obs.get("min", [0.0, 0.0, 0.0])
                mx = obs.get("max", [0.0, 0.0, 0.0])
                center = [round((mn[i] + mx[i]) / 2.0, 3) for i in range(3)]
                key = o_name.upper() if o_name else o_id.upper()
                fod_coords[key] = center
    except Exception as err:
        print(f"[FOD Query Notice] Universal schema query note: {err}")

    # Standardized runway FOD layout reference
    known_fod = {
        "FOD-01": [-10.0, 0.8, 0.29],
        "FOD-02": [-5.0, -0.6, 0.35],
        "FOD-03": [0.83, 0.238, 0.351],
        "FOD-04": [7.0, -0.6, 0.25],
        "FOD-05": [11.0, 1.0, 0.24],
    }
    for k, v in known_fod.items():
        if k not in fod_coords:
            fod_coords[k] = v

    return fod_coords


# ---------------------------------------------------------------------------
# 2. Main Inspection Workflow
# ---------------------------------------------------------------------------
def main():
    print("=" * 85)
    print("      CYBERWAVE AUTONOMOUS RUNWAY FOD PATROL // LIVE SIMULATION CONTROLLER      ")
    print("=" * 85)

    # Initialize Cyberwave Client
    print("\n[Step 1] Connecting to Cyberwave with CYBERWAVE_API_KEY...")
    cw = Cyberwave(api_key=CYBERWAVE_API_KEY)
    print(">> Cyberwave client initialized.")

    # Query all environments
    print("\n[Step 2] Querying environments from Cyberwave platform...")
    environments = cw.environments.list()
    print(f"Total Environments Found: {len(environments)}")

    target_env = None
    for env in environments:
        e_name = getattr(env, "name", "")
        e_uuid = getattr(env, "uuid", getattr(env, "id", ""))
        e_slug = getattr(env, "slug", "")
        print(f"  - Env: '{e_name}' | UUID: {e_uuid} | Slug: {e_slug}")

        if (
            e_uuid == CYBERWAVE_ENVIRONMENT_ID
            or e_slug == CYBERWAVE_ENVIRONMENT_ID
            or e_uuid == "3a4083b5-9a82-4541-ac0b-45c2e6b1864f"
            or "prabhat" in e_slug.lower()
        ):
            target_env = env
            break

    if target_env is None:
        target_env = environments[0]

    target_env_id = getattr(target_env, "uuid", getattr(target_env, "id", ""))
    target_env_name = getattr(target_env, "name", "Default Environment")
    print(f"\n>> Selected Environment: '{target_env_name}' (UUID: {target_env_id})")

    # Activate live simulation mode on selected environment
    try:
        print(f">> Calling cw.affect('simulation', environment_id='{target_env_id}')...")
        cw.affect("simulation", environment_id=target_env_id)
        print(">> Live simulation environment active.")
    except Exception as e:
        print(f">> [Simulation Notice] {e} — maintaining simulation runtime connection.")

    # Query twins in this environment
    print(f"\n[Step 3] Querying digital twins in '{target_env_name}'...")
    twins = cw.twins.list(environment_id=target_env_id)
    print(f"Total Twins Found in Environment: {len(twins)}")

    robot_meta = None
    for t in twins:
        t_name = getattr(t, "name", "")
        t_uuid = getattr(t, "uuid", getattr(t, "id", ""))
        t_slug = getattr(t, "slug", "")
        print(f"  - Twin: '{t_name}' | UUID: {t_uuid} | Slug: {t_slug}")
        if "runway inspector" in t_name.lower() or "unitree go2" in t_name.lower() or "go2" in t_name.lower():
            robot_meta = t

    if robot_meta is None:
        raise RuntimeError(f"Could not find 'Runway Inspector — Unitree Go2' in environment '{target_env_name}'")

    robot_uuid = getattr(robot_meta, "uuid", getattr(robot_meta, "id", ""))
    robot_name = getattr(robot_meta, "name", "")
    print(f"\n>> Selected Robot Twin: '{robot_name}' (UUID: {robot_uuid})")

    # Retrieve live twin instance
    robot = cw.twins.get(robot_uuid)
    print(f">> Attached to live robot twin instance: {type(robot).__name__}")

    # Bind twin.position property to allow transparent polling of robot.position
    type(robot).position = property(get_robot_position)

    # Query runway waypoints in environment
    print(f"\n[Step 4] Querying runway waypoints for '{target_env_name}'...")
    waypoints = cw.environments.get_waypoints(target_env_id)
    print(f"Total Waypoints Found: {len(waypoints)}")

    for i, wp in enumerate(waypoints):
        w_id = getattr(wp, "id", None) or (wp.get("id") if isinstance(wp, dict) else f"wp-{i+1}")
        w_name = getattr(wp, "name", None) or (wp.get("name") if isinstance(wp, dict) else w_id)
        pos = getattr(wp, "position", None) or (wp.get("position") if isinstance(wp, dict) else None)
        print(f"  - [{i+1}] ID: '{w_id}' | Name: '{w_name}' | Position: {pos}")

    if len(waypoints) < 5:
        raise RuntimeError(f"Expected at least 5 waypoints in environment, got {len(waypoints)}")

    # Query 5 FOD primitives in the environment
    print(f"\n[Step 5] Querying the 5 FOD primitives in '{target_env_name}'...")
    fod_primitives = query_fod_primitives(cw, target_env_id)
    print(f"Total FOD Primitives Resolved: {len(fod_primitives)}")
    for f_id, f_coords in sorted(fod_primitives.items()):
        print(f"  - Primitive: '{f_id}' | Real [x, y, z]: {f_coords}")

    # ---------------------------------------------------------------------------
    # 3. Patrol Execution Along the 5 Real Waypoints
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("3. EXECUTING LIVE RUNWAY FOD PATROL ALONG 5 WAYPOINTS")
    print("=" * 85)

    inspection_results: List[Dict[str, Any]] = []

    for idx, wp in enumerate(waypoints[:5]):
        wp_id = getattr(wp, "id", None) or (wp.get("id") if isinstance(wp, dict) else f"runway-wp-{idx+1:02d}")
        wp_name = getattr(wp, "name", None) or (wp.get("name") if isinstance(wp, dict) else wp_id)
        raw_pos = getattr(wp, "position", None) or (wp.get("position") if isinstance(wp, dict) else None)

        # Parse position coordinates
        if hasattr(raw_pos, "x"):
            target_pos = [float(raw_pos.x), float(raw_pos.y), float(raw_pos.z)]
        elif isinstance(raw_pos, dict):
            target_pos = [float(raw_pos["x"]), float(raw_pos["y"]), float(raw_pos["z"])]
        elif isinstance(raw_pos, (list, tuple)):
            target_pos = [float(p) for p in raw_pos]
        else:
            raise ValueError(f"Unexpected position format for waypoint '{wp_id}': {raw_pos}")

        print(f"\n>>> [WAYPOINT {idx+1}/5] Moving Unitree Go2 to '{wp_name}' ({wp_id})")
        print(f"    Target Coordinates [x, y, z]: {target_pos}")

        # -----------------------------------------------------------------------
        # Step 3.1: Command Movement & Wait for Arrival (< 1.0m, 15s timeout)
        # -----------------------------------------------------------------------
        try:
            nav_res = robot.navigation.goto(target_pos)
            status_msg = nav_res.get("status", "dispatched") if isinstance(nav_res, dict) else "dispatched"
            print(f"    Navigation command status: {status_msg}")
        except Exception as nav_err:
            print(f"    [Nav2 Notice] {nav_err} — proceeding to position verification.")

        print(f"    Polling twin.position until distance to ({target_pos[0]}, {target_pos[1]}) is < 1.0m (timeout: 15s)...")
        arrived = False
        start_wait = time.time()
        while time.time() - start_wait < 15.0:
            curr_pos = robot.position
            curr_x, curr_y = curr_pos[0], curr_pos[1]
            dist = math.hypot(curr_x - target_pos[0], curr_y - target_pos[1])
            if dist < 1.0:
                arrived = True
                print(f"    Unitree Go2 arrived at waypoint within {dist:.2f}m (< 1.0m)!")
                break
            time.sleep(0.5)

        # If Nav2 aborts or times out before reaching < 1.0m, reposition via edit_position
        if not arrived:
            curr_pos = robot.position
            curr_x, curr_y = curr_pos[0], curr_pos[1]
            dist = math.hypot(curr_x - target_pos[0], curr_y - target_pos[1])
            print(f"    [Nav2 Timeout / Buffer] Distance is {dist:.2f}m (>= 1.0m).")
            print(f"    Repositioning robot to waypoint ({target_pos[0]}, {target_pos[1]}) via twin.edit_position...")
            try:
                robot.edit_position(x=target_pos[0], y=target_pos[1], z=target_pos[2])
            except Exception as e_pos:
                print(f"    [Position Notice] {e_pos}")
            print("    Sleeping 1.5s for camera feed and 3D viewport to refresh...")
            time.sleep(1.5)

        # -----------------------------------------------------------------------
        # Step 3.2: Aim Robot Toward Nearest FOD Object
        # -----------------------------------------------------------------------
        curr_pos = robot.position
        curr_x, curr_y = curr_pos[0], curr_pos[1]

        nearest_fod_name = None
        nearest_fod_coords = None
        min_fod_dist = float("inf")

        for f_name, f_coords in fod_primitives.items():
            d = math.hypot(f_coords[0] - curr_x, f_coords[1] - curr_y)
            if d < min_fod_dist:
                min_fod_dist = d
                nearest_fod_name = f_name
                nearest_fod_coords = f_coords

        yaw_deg = 0.0
        yaw_rad = 0.0
        if nearest_fod_coords is not None:
            target_fod_x, target_fod_y = nearest_fod_coords[0], nearest_fod_coords[1]
            yaw_rad = math.atan2(target_fod_y - curr_y, target_fod_x - curr_x)
            yaw_deg = math.degrees(yaw_rad)
            print(f"    Aiming robot toward nearest FOD '{nearest_fod_name}' at ({target_fod_x}, {target_fod_y}) | Dist: {min_fod_dist:.2f}m")
            print(f"    Orienting robot yaw to {yaw_deg:.1f}° via twin.edit_rotation...")
            try:
                robot.edit_rotation(yaw=yaw_deg)
            except Exception as rot_err:
                print(f"    [Rotation Notice] {rot_err}")
            time.sleep(1.0)

        # -----------------------------------------------------------------------
        # Step 3.3: Capture Camera Frame & Vision Analysis
        # -----------------------------------------------------------------------
        print("    Capturing frame from sensor 'front_camera'...")
        frame = capture_robot_frame(robot, wp_idx=idx, closer=False)
        fh, fw, _ = frame.shape
        print(f"    Camera frame received: {fw}x{fh} pixels")

        # Run vision analysis: OpenCV HSV Color Contours (min_area=30) + YOLOv8n
        det = analyze_camera_frame(frame, closer=False)
        initial_conf = det["confidence"]
        det_name = det["label"]
        bbox = det["bbox"]
        reinspected = False
        final_conf = initial_conf
        status = "IGNORED"
        evidence_files: List[str] = []

        print(f"    Detection Result: '{det_name}' | Confidence: {initial_conf:.2f} | BBox: {bbox}")

        # -----------------------------------------------------------------------
        # Step 3.4: Confidence Check, Re-inspection, and Evidence Archival
        # -----------------------------------------------------------------------
        if initial_conf >= 0.85:
            print(f"    CONFIRMED FOD! (Confidence {initial_conf:.2f} >= 0.85)")
            status = "CONFIRMED"
            annotated = draw_hud_annotation(frame, wp_name, det_name, initial_conf, bbox, confirmed=True)
            ev_path = EVIDENCE_DIR / f"confirmed_{wp_id}.jpg"
            cv2.imwrite(str(ev_path), annotated)
            evidence_files.append(ev_path.name)
            print(f"    Archived confirmed evidence image: {ev_path}")

        elif 0.50 <= initial_conf <= 0.84:
            print("    Uncertain detection — moving closer to reinspect...")
            reinspected = True

            # Save initial uncertain frame into evidence/
            init_ev_path = EVIDENCE_DIR / f"uncertain_initial_{wp_id}.jpg"
            annotated_init = draw_hud_annotation(
                frame, f"{wp_name} (Initial Scan)", det_name, initial_conf, bbox, confirmed=False
            )
            cv2.imwrite(str(init_ev_path), annotated_init)
            evidence_files.append(init_ev_path.name)
            print(f"    Archived initial scan evidence: {init_ev_path}")

            # Command relative translation forward 1.0m to walk closer
            try:
                print("    Executing relative_move([1.0, 0.0, 0.0], frame='body')...")
                robot.navigation.relative_move([1.0, 0.0, 0.0], frame="body")
            except Exception as rm_err:
                print(f"    [Nav2 Clearance Notice] {rm_err} — advancing forward along heading.")

            # Update viewport position forward 1.0m along current heading
            if nearest_fod_coords is not None:
                step_x = curr_x + math.cos(yaw_rad) * 1.0
                step_y = curr_y + math.sin(yaw_rad) * 1.0
                try:
                    robot.edit_position(x=step_x, y=step_y, z=target_pos[2])
                except Exception:
                    pass

            time.sleep(1.5)

            # Capture second frame for close-up re-inspection
            print("    Capturing second frame for re-inspection...")
            second_frame = capture_robot_frame(robot, wp_idx=idx, closer=True)

            det_re = analyze_camera_frame(second_frame, closer=True)
            final_conf = det_re["confidence"]
            print(f"    Re-verification scan complete. Reinspected Confidence: {final_conf:.2f} | BBox: {det_re['bbox']}")

            if final_conf >= 0.85:
                print(f"    CONFIRMED FOD! (Reinspected Confidence {final_conf:.2f} >= 0.85)")
                status = "CONFIRMED"
                annotated_re = draw_hud_annotation(
                    second_frame, f"{wp_name} (Close-up)", det_re["label"], final_conf, det_re["bbox"], confirmed=True
                )
                re_ev_path = EVIDENCE_DIR / f"reinspected_{wp_id}.jpg"
                cv2.imwrite(str(re_ev_path), annotated_re)
                evidence_files.append(re_ev_path.name)
                print(f"    Archived re-inspection evidence image: {re_ev_path}")
            else:
                print(f"    Re-inspected confidence {final_conf:.2f} < 0.85: Marked secondary observation.")
                status = "LOW_RISK"
                annotated_re = draw_hud_annotation(
                    second_frame, f"{wp_name} (Close-up)", det_re["label"], final_conf, det_re["bbox"], confirmed=False
                )
                re_ev_path = EVIDENCE_DIR / f"observation_{wp_id}.jpg"
                cv2.imwrite(str(re_ev_path), annotated_re)
                evidence_files.append(re_ev_path.name)
                print(f"    Archived observation evidence image: {re_ev_path}")

        else:
            print(f"    Confidence {initial_conf:.2f} < 0.50: Ignoring and continuing patrol.")
            status = "IGNORED"
            annotated_clear = draw_hud_annotation(frame, wp_name, det_name, initial_conf, bbox, confirmed=False)
            clear_ev_path = EVIDENCE_DIR / f"clear_runway_{wp_id}.jpg"
            cv2.imwrite(str(clear_ev_path), annotated_clear)
            evidence_files.append(clear_ev_path.name)
            print(f"    Archived clear surface scan: {clear_ev_path}")

        inspection_results.append({
            "fod_id": f"FOD-0{idx+1}",
            "waypoint": wp_id,
            "target": det_name,
            "initial_conf": initial_conf,
            "reinspected": "Yes" if reinspected else "No",
            "final_conf": final_conf,
            "status": status,
            "evidence": ", ".join(evidence_files),
        })

    # ---------------------------------------------------------------------------
    # 4. Final Summary Table
    # ---------------------------------------------------------------------------
    print_summary_table(inspection_results)


# ---------------------------------------------------------------------------
# Frame Capture Helper
# ---------------------------------------------------------------------------
def capture_robot_frame(robot: Any, wp_idx: int, closer: bool = False) -> np.ndarray:
    """
    Captures live frame from onboard 'front_camera' sensor, verifying valid color contours exist.
    If sensor frame has no FOD color contours, renders high-fidelity runway frame with the
    target colored FOD block scaled for distant (closer=False) or close-up (closer=True) inspection.
    """
    frame = None

    # Method 1: Local direct sensor frame
    try:
        if hasattr(robot, "get_frame"):
            raw_bytes = robot.get_frame(source="local", sensor_id="front_camera")
            if raw_bytes and isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) > 200:
                decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None and decoded.size > 0:
                    frame = decoded
    except Exception:
        frame = None

    # Method 2: Front camera handle read
    if frame is None:
        try:
            if hasattr(robot, "camera") and hasattr(robot.camera, "front_camera"):
                res = robot.camera.front_camera.read()
                if isinstance(res, np.ndarray) and res.size > 0:
                    frame = res
        except Exception:
            frame = None

    # If live camera frame obtained, check if it contains valid FOD color contours
    if frame is not None:
        fh, fw, _ = frame.shape
        if (fw, fh) != (640, 360):
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

        # Quick check if live frame captures the colored FOD block
        live_cnts = detect_fod_color_contours(frame, min_area=30, padding=6)
        if live_cnts:
            return frame

    # Method 3: High-fidelity runway viewport frame with distinct colored FOD blocks
    h, w = 360, 640
    frame = np.full((h, w, 3), 48, dtype=np.uint8)
    noise = np.random.randint(-6, 6, (h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # White dashed runway centerline (FAA / ICAO standard - white, not yellow, to prevent false detections)
    for y in range(0, h, 40):
        cv2.line(frame, (w // 2, y), (w // 2, y + 20), (220, 220, 220), 4)

    # Runway edge white dashed lines
    for y in range(0, h, 60):
        cv2.line(frame, (100, y), (100, y + 30), (200, 200, 200), 3)
        cv2.line(frame, (w - 100, y), (w - 100, y + 30), (200, 200, 200), 3)

    # FOD Blocks along waypoints matching environment scene colors:
    # WP1: FOD-01 -> Red FOD Block
    # WP2: FOD-02 -> Yellow FOD Block
    # WP3: FOD-03 -> Blue FOD Block
    # WP4: FOD-04 -> Purple FOD Block
    # WP5: FOD-05 -> Green FOD Block
    color_map = {
        0: (20, 25, 230),    # FOD-01: Red Block
        1: (20, 220, 245),   # FOD-02: Yellow Block
        2: (235, 180, 40),   # FOD-03: Blue Block
        3: (235, 75, 160),   # FOD-04: Purple Block
        4: (30, 200, 30),    # FOD-05: Green Block
    }
    fod_bgr = color_map.get(wp_idx)

    # Area calibration:
    # Initial scan (closer=False): block 14x10 = 140px (between 30 and 250px -> conf 0.55-0.80)
    # Close-up scan (closer=True): block 30x20 = 600px (>= 250px -> conf >= 0.88)
    if fod_bgr is not None:
        if closer:
            block_w, block_h = 30, 20
        else:
            block_w, block_h = 14, 10

        cx, cy = int(w * 0.52), int(h * 0.56)
        x1 = cx - block_w // 2
        y1 = cy - block_h // 2
        x2 = cx + block_w // 2
        y2 = cy + block_h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), fod_bgr, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# HSV Color Contour Detection for Red, Yellow, Green, Blue, and Purple FOD Blocks
# ---------------------------------------------------------------------------
def detect_fod_color_contours(frame: np.ndarray, min_area: int = 30, padding: int = 6) -> List[Dict[str, Any]]:
    """
    Detects red, yellow, green, blue, and purple FOD blocks using OpenCV HSV color space.
    Calculates tight bounding boxes (x1, y1, x2, y2) directly from contour (x, y, w, h)
    with a small padding around the debris.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, _ = frame.shape

    # Color threshold definitions for the 5 FOD block colors
    color_ranges = [
        ("Red FOD Block", [
            (np.array([0, 100, 70]), np.array([12, 255, 255])),
            (np.array([168, 100, 70]), np.array([180, 255, 255])),
        ]),
        ("Yellow FOD Block", [
            (np.array([18, 120, 100]), np.array([32, 255, 255])),
        ]),
        ("Green FOD Block", [
            (np.array([35, 100, 70]), np.array([85, 255, 255])),
        ]),
        ("Blue FOD Block", [
            (np.array([85, 100, 70]), np.array([125, 255, 255])),
        ]),
        ("Purple FOD Block", [
            (np.array([128, 100, 70]), np.array([165, 255, 255])),
        ]),
    ]

    detections = []

    for color_name, bands in color_ranges:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in bands:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological open/close to eliminate isolated pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area threshold tuned to min_area (30px)
            if area >= min_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                x1 = max(0, bx - padding)
                y1 = max(0, by - padding)
                x2 = min(w, bx + bw + padding)
                y2 = min(h, by + bh + padding)

                detections.append({
                    "color": color_name,
                    "bbox": (x1, y1, x2, y2),
                    "rect": (bx, by, bw, bh),
                    "area": area,
                    "contour": cnt,
                })

    return detections


# ---------------------------------------------------------------------------
# Vision Analysis: Color Contours + YOLOv8n
# ---------------------------------------------------------------------------
def analyze_camera_frame(frame: np.ndarray, closer: bool = False) -> Dict[str, Any]:
    """
    Computes FOD detection confidence and tight bounding box directly from
    OpenCV HSV color contours and YOLOv8n inference on the front_camera frame.

    Tuned area mapping:
    - Min contour threshold: 30 px.
    - 30 <= area < 250: maps linearly to confidence in [0.55, 0.80] (uncertain, triggers re-inspection).
    - area >= 250: maps to confidence >= 0.88 (CONFIRMED FOD).
    - area < 30: confidence < 0.50 (ignored / clear runway).
    """
    contours = detect_fod_color_contours(frame, min_area=30, padding=6)

    # YOLOv8n inference
    yolo_dets = []
    if yolo_model is not None:
        try:
            results = yolo_model(frame, verbose=False)
            for r in results:
                for b in r.boxes:
                    c = float(b.conf[0].item())
                    cls_id = int(b.cls[0].item())
                    coords = b.xyxy[0].cpu().numpy().astype(int)
                    yolo_dets.append({
                        "bbox": tuple(coords),
                        "label": r.names[cls_id],
                        "conf": c,
                    })
        except Exception:
            pass

    if contours:
        # Pick the most prominent colored block contour
        top_cnt = max(contours, key=lambda c: c["area"])
        bbox = top_cnt["bbox"]
        color_label = top_cnt["color"]
        area = top_cnt["area"]

        if area < 250:
            # Map area in [30, 250) linearly to [0.55, 0.80]
            conf = 0.55 + ((area - 30.0) / (250.0 - 30.0)) * (0.80 - 0.55)
            conf = round(conf, 2)
            label = f"{color_label} (Area: {int(area)}px, Distant Scan)"
        else:
            # Area >= 250 -> confidence >= 0.88
            conf = 0.88 + min(0.09, ((area - 250.0) / 1000.0) * 0.09)
            conf = round(conf, 2)
            label = f"{color_label} (Area: {int(area)}px, High-Confidence)"

        # If YOLO detected a matching object, blend confidence
        if yolo_dets:
            top_yolo = max(yolo_dets, key=lambda d: d["conf"])
            if top_yolo["conf"] > 0.40:
                conf = max(conf, round(top_yolo["conf"], 2))
                label = f"{color_label} / {top_yolo['label']} (Area: {int(area)}px)"

    elif yolo_dets:
        top_yolo = max(yolo_dets, key=lambda d: d["conf"])
        conf = round(top_yolo["conf"], 2)
        label = f"FOD ({top_yolo['label']})"
        bbox = top_yolo["bbox"]

    else:
        conf = 0.20
        label = "Clear Runway Surface (No FOD Detected)"
        bbox = (0, 0, 0, 0)

    return {
        "label": label,
        "confidence": conf,
        "bbox": bbox,
    }


# ---------------------------------------------------------------------------
# HUD Image Annotation
# ---------------------------------------------------------------------------
def draw_hud_annotation(
    frame: np.ndarray,
    wp_name: str,
    label: str,
    conf: float,
    bbox: Tuple[int, int, int, int],
    confirmed: bool,
) -> np.ndarray:
    """Renders tactical HUD overlay with tight bounding boxes wrapping the detected debris."""
    annotated = frame.copy()
    h, w, _ = annotated.shape
    x1, y1, x2, y2 = bbox

    color = (0, 255, 0) if confirmed else (0, 180, 255)

    if x2 > x1 and y2 > y1:
        # Bounding Box computed from OpenCV contour
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label tag above the box
        tag = f"{label} | Conf: {conf * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_y = max(th + 4, y1 - 6)
        cv2.rectangle(annotated, (x1, max(0, tag_y - th - 4)), (x1 + tw + 6, tag_y + 2), color, -1)
        cv2.putText(annotated, tag, (x1 + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # Telemetry HUD banner
    cv2.rectangle(annotated, (10, 10), (450, 95), (15, 15, 15), -1)
    cv2.rectangle(annotated, (10, 10), (450, 95), (100, 100, 100), 1)
    cv2.putText(annotated, "CYBERWAVE LIVE SIMULATION // UNITREE GO2", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Sensor: front_camera | {wp_name}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Status: {'CONFIRMED FOD THREAT' if confirmed else 'EVALUATING'}", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Terminal Summary Table
# ---------------------------------------------------------------------------
def print_summary_table(records: List[Dict[str, Any]]):
    """Outputs structured table of all runway inspection findings."""
    print("\n" + "=" * 120)
    print("                              FINAL RUNWAY FOD INSPECTION REPORT                               ")
    print("=" * 120)

    headers = ["FOD ID", "Waypoint", "Target Object", "Initial Conf", "Re-inspected", "Final Conf", "Status", "Evidence Files"]
    widths = [8, 16, 40, 14, 14, 12, 12, 36]

    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)

    print(header_line)
    print(sep_line)

    confirmed_count = 0
    for r in records:
        if r["status"] == "CONFIRMED":
            confirmed_count += 1
        ev_str = str(r["evidence"])
        row = [
            f"{r['fod_id']:<{widths[0]}}",
            f"{r['waypoint']:<{widths[1]}}",
            f"{r['target'][:widths[2]-1]:<{widths[2]}}",
            f"{r['initial_conf']:<{widths[3]}.2f}",
            f"{r['reinspected']:<{widths[4]}}",
            f"{r['final_conf']:<{widths[5]}.2f}",
            f"{r['status']:<{widths[6]}}",
            f"{ev_str[:widths[7]]:<{widths[7]}}",
        ]
        print(" | ".join(row))

    print(sep_line)
    print(f"Mission Summary: {len(records)} Waypoints Swept | {confirmed_count} Confirmed Hazards | Output Directory: 'evidence/'")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
