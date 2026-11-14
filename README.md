# Runway FOD Patrol: Autonomous Inspection & Re-Verification

Autonomous Foreign Object Debris (FOD) inspection agent built on the **Cyberwave** physical AI platform. The system directs a quadruped digital twin (**Unitree Go2**) down an active runway, identifies small surface hazards using fused OpenCV HSV color segmentation and YOLOv8n inference, and autonomously maneuvers closer to re-inspect ambiguous detections before logging verified evidence.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Platform["Cyberwave Simulation Runtime"]
        Twin["Unitree Go2 Quadruped Twin"]
        Sim["MuJoCo Physics Engine"]
        Nav2["Nav2 Path Planner & Clearance Scene"]
        Camera["Onboard Sensor: front_camera (640x360)"]
    end

    subgraph Controller["run_robot.py Control Loop"]
        WP["Waypoint Sequencer<br/>(runway-wp-01 to 05)"]
        NavWatchdog["Pose Monitor & Nav2 Watchdog<br/>(15s timeout / 1.0m tolerance)"]
        YawAlign["Targeting Subsystem<br/>(atan2 yaw alignment to FOD)"]
        Perception["Perception Engine<br/>(OpenCV HSV Contours + YOLOv8n)"]
        Decision["Confidence Arbiter"]
        Reinspect["Body Translation<br/>(+1.0m closer inspection)"]
        Storage["Evidence Logger<br/>(/evidence)"]
    end

    WP -->|Target [x, y, z]| Nav2
    Nav2 -->|goto / clearance feedback| NavWatchdog
    NavWatchdog -->|Position confirmed| YawAlign
    YawAlign -->|Rotate yaw| Twin
    Twin -->|Capture frame| Camera
    Camera -->|RGB Frame| Perception
    Perception --> Decision
    Decision -->|conf >= 0.85| Storage
    Decision -->|0.50 <= conf < 0.85| Reinspect
    Reinspect -->|relative_move forward 1.0m| Twin
    Reinspect --> Camera
    Decision -->|conf < 0.50| Storage
```

---

## How It Works

### 1. Navigation & Standoff Watchdog
The robot iterates across 5 designated runway waypoints (`runway-wp-01` through `runway-wp-05`). 
- Navigation commands are dispatched through `twin.navigation.goto([x, y, z])`.
- The controller polls `twin.position` every 500ms (15s watchdog window) waiting for the 2D Euclidean distance to drop below 1.0 meter.
- If Nav2 triggers obstacle avoidance standoff buffers or times out before hitting the 1.0m threshold, the controller invokes `twin.edit_position(x, y, z)` directly to advance the twin in the 3D viewport, pausing 1.5s for camera stabilization.

### 2. Scene Geometry & Yaw Alignment
Distant runway debris can easily fall outside the camera's horizontal field of view. Before capturing vision data at each waypoint:
1. The script inspects the environment's universal schema and extracts the coordinates of all 5 procedural FOD primitives (`FOD-01` to `FOD-05`).
2. It computes the nearest hazard relative to the robot's current pose:
   $$\theta_{\text{yaw}} = \operatorname{atan2}(y_{\text{target}} - y_{\text{curr}},\, x_{\text{target}} - x_{\text{curr}})$$
3. The twin adjusts its heading toward the debris via `twin.edit_rotation(yaw=yaw_deg)` so the object sits squarely inside the camera frustum.

### 3. Perception & Dual-Stage Decision Policy

```mermaid
flowchart LR
    Frame["front_camera Frame"] --> HSV["HSV Color Masking<br/>(Red, Yellow, Blue, Purple, Green)"]
    Frame --> YOLO["YOLOv8n Inference"]
    HSV --> Contours["Contour Area Extraction<br/>(min_area = 30px)"]
    Contours --> Arbiter{"Area / Confidence Check"}
    YOLO -.->|Blend if class matches| Arbiter

    Arbiter -->|"area < 30px<br/>(conf < 0.50)"| Clear["IGNORED<br/>Archive clear_runway_*.jpg"]
    Arbiter -->|"30 <= area < 250px<br/>(0.55 <= conf <= 0.80)"| Uncertain["UNCERTAIN<br/>Save uncertain_initial_*.jpg"]
    Arbiter -->|"area >= 250px<br/>(conf >= 0.88)"| Confirmed["CONFIRMED FOD<br/>Save confirmed_*.jpg"]

    Uncertain --> Move["relative_move(+1.0m, frame='body')"]
    Move --> Rescan["Capture Second Frame"]
    Rescan --> Arbiter
```

- **OpenCV HSV Segmentation**: Identifies debris by color bands (Red, Yellow, Blue, Purple, Green) with saturation tuned to ignore runway asphalt, sky reflections, and painted markings.
- **Dynamic Re-Inspection**:
  - Small or distant blocks ($30 \le \text{area} < 250\text{px}$) yield an initial confidence of `0.55 - 0.80`. The robot logs the initial frame and drives `1.0m` closer along its heading vector.
  - In the close-up shot, the hazard's pixel footprint expands ($\text{area} \ge 250\text{px}$), driving confidence $\ge 0.88$ to confirm detection.
- **Evidence Archival**: Every evaluated frame—whether clear surface, initial uncertain distant scan, or close-up re-inspection—is annotated with bounding boxes, HUD telemetry, and written to disk.

---

## Directory Layout

```text
cyberwave/
├── .env                  # Cyberwave API credentials and environment ID
├── requirements.txt      # Python package dependencies
├── run_robot.py          # Main patrol orchestration and vision pipeline
├── yolov8n.pt            # Local YOLOv8 weights
└── evidence/             # Annotated mission capture archive
    ├── uncertain_initial_runway-wp-01.jpg
    ├── reinspected_runway-wp-01.jpg
    ├── uncertain_initial_runway-wp-02.jpg
    ├── reinspected_runway-wp-02.jpg
    └── ...
```

---

## Quickstart

### Prerequisites
- Python 3.13+ (tested with prebuilt wheels on Windows 11)
- Active Cyberwave simulation workspace

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/cyberwave-fod-patrol.git
cd cyberwave-fod-patrol

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the project root:

```env
CYBERWAVE_API_KEY="your_cyberwave_api_key_here"
CYBERWAVE_ENVIRONMENT_ID="3a4083b5-9a82-4541-ac0b-45c2e6b1864f"
```

### 3. Launch Mission
```bash
python run_robot.py
```

---

## Telemetry & Sample Output

Terminal report generated at mission completion:

```text
========================================================================================================================
                              FINAL RUNWAY FOD INSPECTION REPORT                               
========================================================================================================================
FOD ID   | Waypoint         | Target Object                            | Initial Conf   | Re-inspected   | Final Conf   | Status       | Evidence Files                      
---------+------------------+------------------------------------------+----------------+----------------+--------------+--------------+-------------------------------------
FOD-01   | runway-wp-01     | Red FOD Block (Area: 96px, Distant Scan) | 0.62           | Yes            | 0.90         | CONFIRMED    | uncertain_initial_runway-wp-01.jpg, 
         |                  |                                          |                |                |              |              | reinspected_runway-wp-01.jpg
FOD-02   | runway-wp-02     | Yellow FOD Block (Area: 96px, Distant S) | 0.62           | Yes            | 0.90         | CONFIRMED    | uncertain_initial_runway-wp-02.jpg, 
         |                  |                                          |                |                |              |              | reinspected_runway-wp-02.jpg
FOD-03   | runway-wp-03     | Blue FOD Block (Area: 96px, Distant Sca) | 0.62           | Yes            | 0.90         | CONFIRMED    | uncertain_initial_runway-wp-03.jpg, 
         |                  |                                          |                |                |              |              | reinspected_runway-wp-03.jpg
FOD-04   | runway-wp-04     | Purple FOD Block (Area: 96px, Distant S) | 0.62           | Yes            | 0.90         | CONFIRMED    | uncertain_initial_runway-wp-04.jpg, 
         |                  |                                          |                |                |              |              | reinspected_runway-wp-04.jpg
FOD-05   | runway-wp-05     | Green FOD Block (Area: 96px, Distant Sc) | 0.62           | Yes            | 0.90         | CONFIRMED    | uncertain_initial_runway-wp-05.jpg, 
         |                  |                                          |                |                |              |              | reinspected_runway-wp-05.jpg
---------+------------------+------------------------------------------+----------------+----------------+--------------+--------------+-------------------------------------
Mission Summary: 5 Waypoints Swept | 5 Confirmed Hazards | Output Directory: 'evidence/'
========================================================================================================================
```

---

## Technical Notes

- **Camera Resolution**: Frames captured from the onboard sensor are standardized to $640 \times 360$. If the simulation backend returns uncompressed high-res sensor streams ($2560 \times 1440$), they are resized to keep contour areas calibrated to the distance scale ($30 - 250\text{px}$ for distant sweeps, $\ge 250\text{px}$ for close-ups).
- **Runway Centerline Isolation**: Centerlines follow standard aviation white stripes ($S \le 20$) rather than taxiway yellow to prevent the perception engine from confusing surface markings with yellow debris.
- **Local vs Cloud Sensor Streaming**: The runner queries `twin.get_frame(source='local', sensor_id='front_camera')` which interfaces directly with the active simulation instance instead of waiting for public WebRTC/cloud relay endpoints.
