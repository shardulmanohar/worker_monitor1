import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.dont_write_bytecode = True

import cv2
import time
import numpy as np
import logging
import traceback
import mediapipe as mp
from assets.config_reader import load_config  # Your YAML loader


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def get_valid_roi(frame_shape, config_roi):
    if config_roi and isinstance(config_roi, list) and len(config_roi) >= 3:
        return [tuple(point) for point in config_roi]
    h, w = frame_shape[:2]
    return [(0, 0), (w, 0), (w, h), (0, h)]

def apply_roi_mask_if_needed(frame, roi_points):
    h, w = frame.shape[:2]
    full_frame = [(0, 0), (w, 0), (w, h), (0, h)]
    if roi_points != full_frame:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_polygon = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_polygon], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)
    return frame

def main():
    # === Load Config ===
    config = load_config("assets/config.yaml")
    video_source = config["video_path"] if config["video_path"] else 0
    resize_width = config["resize_width"]
    resize_height = config["resize_height"]
    thresholds = config["thresholds"]
    idle_threshold_seconds = config["idle_threshold_seconds"]
    upright_ref = config["angles"]["upright"]
    slouch_ref = config["angles"]["slouching"]
    roi_from_config = config.get("roi", None)

    # === MediaPipe Setup ===
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    last_positions = {}
    idle_start_time = None
    is_idle = False
    display_name = "Worker Activity Monitor"

    def check_movement(label, idx, landmarks, threshold):
        lm = landmarks[idx]
        if lm.visibility > 0.3:
            curr = np.array([lm.x, lm.y])
            last = last_positions.get(label)
            last_positions[label] = curr
            if last is not None:
                return np.linalg.norm(curr - last) > threshold
            return False
        return None

    def movement_from_pair(label_left, idx_left, label_right, idx_right, threshold):
        move_l = check_movement(label_left, idx_left, landmarks, threshold)
        move_r = check_movement(label_right, idx_right, landmarks, threshold)

        if move_l is not None or move_r is not None:
            if move_l or move_r:
                return "Working"
            else:
                return "Idle"
        return None
    def get_facial_angles_vector(landmarks):
        def angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
        try:
            pts = [
                (landmarks[mp_pose.PoseLandmark.LEFT_EAR], 
                 landmarks[mp_pose.PoseLandmark.LEFT_EYE], 
                 landmarks[mp_pose.PoseLandmark.NOSE]),
                (landmarks[mp_pose.PoseLandmark.LEFT_EYE], 
                 landmarks[mp_pose.PoseLandmark.NOSE], 
                 landmarks[mp_pose.PoseLandmark.RIGHT_EYE]),
                (landmarks[mp_pose.PoseLandmark.NOSE], 
                 landmarks[mp_pose.PoseLandmark.RIGHT_EYE], 
                 landmarks[mp_pose.PoseLandmark.RIGHT_EAR]),
            ]
            return [angle((a.x, a.y), (b.x, b.y), (c.x, c.y)) for a, b, c in pts]
        except:
            return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            print("\n--- DEBUG: Enter key pressed ---")
            # Add debug print logic here

        frame = resize_frame(frame, resize_width, resize_height)
        roi_points = get_valid_roi(frame.shape, roi_from_config)
        frame = apply_roi_mask_if_needed(frame, roi_points)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        status = "No Person"
        status_color = (0, 0, 255)
        current_time = time.time()

        if results.pose_landmarks:
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # === Movement priority: palm → wrist → elbow → shoulder ===
            status = movement_from_pair("palm1_l", mp_pose.PoseLandmark.LEFT_THUMB,
                                        "palm1_r", mp_pose.PoseLandmark.RIGHT_THUMB, thresholds["palm"])

            if status is None:
                status = movement_from_pair("palm2_l", mp_pose.PoseLandmark.LEFT_INDEX,
                                            "palm2_r", mp_pose.PoseLandmark.RIGHT_INDEX, thresholds["palm"])

            if status is None:
                status = movement_from_pair("palm3_l", mp_pose.PoseLandmark.LEFT_PINKY,
                                            "palm3_r", mp_pose.PoseLandmark.RIGHT_PINKY, thresholds["palm"])

            if status is None:
                status = movement_from_pair("wrist_l", mp_pose.PoseLandmark.LEFT_WRIST,
                                            "wrist_r", mp_pose.PoseLandmark.RIGHT_WRIST, thresholds["wrist"])

            if status is None:
                status = movement_from_pair("elbow_l", mp_pose.PoseLandmark.LEFT_ELBOW,
                                            "elbow_r", mp_pose.PoseLandmark.RIGHT_ELBOW, thresholds["elbow"])
            if status is None:
                status = movement_from_pair("shoulder_l", mp_pose.PoseLandmark.LEFT_SHOULDER,
                                            "shoulder_r", mp_pose.PoseLandmark.RIGHT_SHOULDER, thresholds["shoulder"])
            
                # If shoulders are idle or not visible, use posture fallback
                if status is None or status == "Idle":
                    facial_angles = get_facial_angles_vector(landmarks)
                
                    if facial_angles:
                        upright_ref = [124.1, 117.4, 125.8]
                        slouch_ref = [143.1, 94.4, 142.7]
                
                        upright_dist = sum(abs(f - u) for f, u in zip(facial_angles, upright_ref))
                        slouch_dist = sum(abs(f - s) for f, s in zip(facial_angles, slouch_ref))
                
                        if slouch_dist < upright_dist:
                            status = "Idle"
                            status_color = (0, 0, 255)
                        else:
                            status = "Working"
                            status_color = (0, 255, 0)
                
            # === Idle time logic ===
            if status == "Working":
                idle_start_time = None
                is_idle = False
                status_color = (0, 255, 0)
            elif status == 'Idle':
                if idle_start_time is None:
                    idle_start_time = current_time
        # === Display ===
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if idle_start_time:
            idle_time = time.time() - idle_start_time
            cv2.putText(frame, f"Idle Time: {idle_time:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        
        cv2.imshow(display_name, frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program terminated by keyboard interrupt")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}")
        traceback.print_exc()
