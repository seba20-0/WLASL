import os
import json

# Gesture code to name mapping
GESTURE_MAP = {
    "G01": "Hi",
    "G02": "Please",
    "G03": "What?",
    "G04": "Arabic",
    "G05": "University",
    "G06": "You",
    "G07": "Eat",
    "G08": "Sleep",
    "G09": "Go",
    "G10": "UAE"
}

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

# Find all users
users = [d for d in os.listdir(DATASET_ROOT) if d.startswith("user") and os.path.isdir(os.path.join(DATASET_ROOT, d))]

# Build gesture structure
gesture_dict = {gname: [] for gname in GESTURE_MAP.values()}

for user in users:
    user_path = os.path.join(DATASET_ROOT, user)
    for gesture_code, gesture_name in GESTURE_MAP.items():
        gesture_path = os.path.join(user_path, gesture_code)
        if not os.path.isdir(gesture_path):
            continue
        # Each repetition is a folder, video is userXX/GXX/RYY.mp4
        for rep in [d for d in os.listdir(gesture_path) if d.startswith("R")]:
            rep_path = os.path.join(gesture_path, rep)
            # Video file: userXX/GXX/RYY.mp4
            video_filename = rep if rep.endswith('.mp4') else f"{rep}.mp4"
            video_file_rel = os.path.join('MLR511-ArabicSignLanguage-Dataset-MP4', user, gesture_code, video_filename)
            video_id = f"{user}_{gesture_code}_{rep}"
            instance = {
                "video_id": video_id,
                "url": video_file_rel,
                "frame_start": 0,
                "frame_end": 0
            }
            gesture_dict[gesture_name].append(instance)

# Convert to WLASL-style JSON
json_data = []
for gesture_name, instances in gesture_dict.items():
    if instances:
        json_data.append({
            "label": gesture_name,
            "instances": instances
        })

with open("arabic_sign_language_dataset.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"JSON file created with {len(json_data)} gestures.")
