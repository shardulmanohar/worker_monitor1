**⚠️ Requires Python 3.10.16**
# 🧠 Worker Activity Monitoring System

A real-time, lightweight, standalone vision-based system to detect if a worker is **actively engaged**, **idle**, or **slouching**, using **MediaPipe Pose Estimation**. This system is optimized for low-cost hardware and designed for robust, real-world deployment.

---

## 🚀 Features

- 🧍 Pose-based movement detection using MediaPipe
- 🎯 Joint-specific motion thresholds (palms, wrists, elbows, shoulders)
- 🧠 Posture classification using facial landmark angles
- ⏱️ Idle state tracking with customizable timeout
- 🧭 Region of Interest (ROI) masking (configurable)
- ⚡ Optimized for CPU-only environments (Raspberry Pi & laptops)
- 🧰 YAML-based configuration for full customizability
- 🔌 Auto webcam fallback if video path is not given

---

## 📁 Project Structure

```
worker-monitor/
├── assets/
│   ├── config_reader.py       # Loads and validates config.yaml
│   ├── config.yaml            # All settings in one place
├── main.py                    # Main application script
├── requirements.txt           # Python dependencies
└── README.md                  # You're here
```

---

## ⚙️ Configuration (`assets/config.yaml`)

```yaml
video_path: ""  # leave blank to use webcam

resize_width: 640
resize_height: 360

thresholds:
  palm: 0.01
  wrist: 0.01
  elbow: 0.02
  shoulder: 0.03

angles:
  upright: [124.1, 117.4, 125.8]
  slouching: [143.1, 94.4, 142.7]

idle_threshold_seconds: 1

roi:  # Optional. Leave unset for full-frame
  - [0, 0]
  - [640, 0]
  - [640, 360]
  - [0, 360]
```

---

## 🛠️ Configuration Parameters Explained

| Key                       | Type       | Description                                                                 |
|--------------------------|------------|-----------------------------------------------------------------------------|
| `video_path`             | string     | Path to the input video. Leave empty (`""`) to use the default webcam.     |
| `resize_width`           | integer    | Width to resize each frame for processing. Reduces load, improves speed.   |
| `resize_height`          | integer    | Height to resize each frame for processing.                                |
| `thresholds.palm`        | float      | Minimum movement (normalized) to count palm as active. Lower = more sensitive. |
| `thresholds.wrist`       | float      | Movement threshold for wrist joints.                                       |
| `thresholds.elbow`       | float      | Movement threshold for elbows.                                             |
| `thresholds.shoulder`    | float      | Movement threshold for shoulders.                                          |
| `angles.upright`         | list[float]| 3 angles representing upright head posture using facial landmarks.         |
| `angles.slouching`       | list[float]| 3 angles representing slouched head posture using facial landmarks.        |
| `idle_threshold_seconds` | float      | Time (in seconds) before a person is considered idle.                      |
| `roi`                    | list[list] | List of four [x, y] points defining a polygon ROI. Optional. Full frame is used if missing. |

---

## 📦 Installation

```bash
pip install -r requirements.txt
python main.py
```
## 🧠 Detection Logic (How It Works)

The system determines a worker’s activity status using a **hierarchical fallback model**:

1. **Fine joint motion detection (Palms → Wrists)**
   - If either palm or wrist shows movement above the threshold → status is **Working**

2. **Mid joint motion (Elbows)**
   - If wrists are not visible  elbows are checked next

3. **Coarse joint motion (Shoulders)**
   - If elbows are not visible shoulders are used as fallback

4. **Posture estimation via facial angles**
   - If shoulders are also still or invisible, the system uses face landmark angles to determine the pose: (**If elbows or hands are visible pose detection is not used**)
     - **Upright** → Working
     - **Slouching** → Slouching
     - Unclear/low-confidence → Idle

5. **Idle time tracking**
   - If no sufficient activity is detected for `idle_threshold_seconds`, the worker is considered **Idle**

✅ This logic mimics real-world expectations — if finer hand motions can’t be seen, broader body or head posture is used as a fallback.

---

## 💡 Design Choices

### ✅ Why MediaPipe instead of custom TFLite/ONNX model?

MediaPipe Pose is built on top of **TFLite models optimized by Google**, and delivers:
- Real-time, low-latency joint tracking
- Proven efficiency on low-end CPUs
- No training required — ready-to-deploy
> ✅ This satisfies the assignment's instruction to use TFLite or ONNX models while focusing on system design.

---

### ❌ Why Not Grayscale?

Grayscale would reduce input size, but was excluded because:
- MediaPipe Pose **requires RGB input**
- Grayscale would reduce accuracy or cause detection failure

✅ Frame resizing and ROI masking were used instead for optimization.

---

### ❌ Why Not Background Removal?

While background removal is useful visually, it was excluded because:
- It adds heavy computational overhead
- ROI already ensures spatial focus
- Real-time performance was prioritized on limited hardware

---

## 🧠 Future Considerations

If extended for production use, the following could be considered:
- Support for detecting multiple workers simultaneously
- Optional alerting for prolonged inactivity
- Logging and reporting of activity timeline

---

## 📝 License

MIT License — Use it, build on it, make it yours.

---

Made with 🛠️ for vision-based workplace intelligence.
