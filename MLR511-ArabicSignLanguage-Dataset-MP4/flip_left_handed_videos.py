import os
import cv2

# List of users to flip
users_to_flip = ["user01", "user02"]

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

for user in users_to_flip:
    user_path = os.path.join(DATASET_ROOT, user)
    if not os.path.isdir(user_path):
        continue
    for gesture in os.listdir(user_path):
        gesture_path = os.path.join(user_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        for rep in os.listdir(gesture_path):
            rep_path = os.path.join(gesture_path, rep)
            if not os.path.isdir(rep_path):
                continue
            # Video file: userXX/GXX/RYY.mp4
            video_file = os.path.join(gesture_path, f"{rep}.mp4")
            if not os.path.isfile(video_file):
                continue
            print(f"Flipping {video_file} ...")
            # Read video
            cap = cv2.VideoCapture(video_file)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            temp_file = video_file + ".tmp.mp4"
            out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                flipped = cv2.flip(frame, 1)  # Horizontal flip
                out.write(flipped)
            cap.release()
            out.release()
            os.replace(temp_file, video_file)
print("All videos for user01 and user02 have been flipped.")
