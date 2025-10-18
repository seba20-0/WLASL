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
# Define videos folder
VIDEOS_FOLDER = os.path.join(DATASET_ROOT, 'videos')

# List of users for leave-one-out
leave_out_users = ["user01", "user08", "user11"]
all_users = [f"user{str(i).zfill(2)}" for i in range(1, 13)]

# Assign class IDs
gesture_to_id = {name: idx for idx, name in enumerate(GESTURE_MAP.values())}

# Get all video info
video_info = []
for user in all_users:
    for gesture_code, gesture_name in GESTURE_MAP.items():
        for rep in range(1, 11):
            video_id = f"{user}_{gesture_code}_R{str(rep).zfill(2)}"
            video_filename = f"{video_id}.mp4"
            video_file_rel = os.path.join('videos', video_filename)
            video_info.append({
                "video_id": video_id,
                "user": user,
                "gesture": gesture_name,
                "class_id": gesture_to_id[gesture_name],
                "url": video_file_rel
            })

# For each leave-one-out split, generate a JSON
for test_user in leave_out_users:
    split_json = {}
    for info in video_info:
        # Assign subset
        subset = "test" if info["user"] == test_user else "train"
        # Get number of frames
        video_path = os.path.join(VIDEOS_FOLDER, f"{info['video_id']}.mp4")
        if os.path.exists(video_path):
            import cv2
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            num_frames = 0
        split_json[info["video_id"]] = {
            "subset": subset,
            "action": [info["class_id"], 0, num_frames],
            "url": info["url"]
        }
    out_name = f"nslt_leaveout_{test_user}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2, ensure_ascii=False)
    print(f"Generated {out_name} with {len(split_json)} videos.")
