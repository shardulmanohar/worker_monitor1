import yaml
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Set defaults if not specified
    data.setdefault("video_path", "")
    data.setdefault("resize_width", 640)
    data.setdefault("resize_height", 360)
    data.setdefault("idle_threshold_seconds", 6)

    data.setdefault("thresholds", {})
    data["thresholds"].setdefault("palm", 0.01)
    data["thresholds"].setdefault("wrist", 0.01)
    data["thresholds"].setdefault("elbow", 0.02)
    data["thresholds"].setdefault("shoulder", 0.03)

    data.setdefault("angles", {})
    data["angles"].setdefault("upright", [124.1, 117.4, 125.8])
    data["angles"].setdefault("slouching", [143.1, 94.4, 142.7])

    data.setdefault("roi", None)

    return data
