"""
SMART ATTENDANCE SYSTEM (MODULAR + SCALABLE)
-------------------------------------------
Features:
- Face detection + recognition
- No duplicate attendance for the same day
- On-time / Late marking with cutoff time
- Auto-create training folder + CSV
- Enrollment of new students from webcam
- Encoding caching for fast startup
- Clean architecture (easy to add more features)
"""

import cv2
import os
import numpy as np
import pandas as pd
import face_recognition
import pickle
from datetime import datetime, date, time as dtime

# -------------------------
# CONFIGURATION
# -------------------------
TRAIN_DIR = "Training_images"
ENCODING_FILE = "encodings.pkl"
CSV_FILE = "attendance.csv"
CUTOFF_TIME = dtime(9, 30, 0)   # 9:30 AM is ON-TIME cutoff
CAMERA_INDEX = 0


# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def ensure_directories():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(CSV_FILE, index=False)
        print("[INFO] Created attendance.csv")


def load_images():
    names = []
    images = []

    for file in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, file)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(file)[0])
        else:
            print(f"[WARN] Could not read {file}")

    return images, names


def encode_faces(images):
    encoding_list = []

    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if len(encs) > 0:
            encoding_list.append(encs[0])
        else:
            print("[WARN] No face detected in training image")

    return encoding_list


def save_encodings(encodings, names):
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print("[INFO] Saved encodings")


def load_encodings():
    if not os.path.exists(ENCODING_FILE):
        return None
    with open(ENCODING_FILE, "rb") as f:
        return pickle.load(f)


def get_status():
    now = datetime.now().time()
    return "On-Time" if now <= CUTOFF_TIME else "Late"


def already_marked(name):
    df = pd.read_csv(CSV_FILE)
    today = date.today().isoformat()

    data_today = df[df["Date"] == today]
    return not data_today[data_today["Name"] == name].empty


def mark_attendance(name):
    if already_marked(name):
        return

    now = datetime.now()
    row = {
        "Name": name,
        "Date": now.date().isoformat(),
        "Time": now.strftime("%H:%M:%S"),
        "Status": get_status()
    }
    df = pd.DataFrame([row])
    df.to_csv(CSV_FILE, mode="a", index=False, header=False)

    print(f"[MARKED] {name} → {row['Status']} at {row['Time']}")


def enroll_from_webcam(name):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"[ENROLL] Capturing 3 images for {name}. Look toward the camera...")

    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Enroll", frame)
        cv2.waitKey(1000)

        filename = os.path.join(TRAIN_DIR, f"{name}_{i+1}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[SAVED] {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Rebuilding encodings...")
    images, names = load_images()
    encs = encode_faces(images)
    save_encodings(encs, names)


# -------------------------
# MAIN FACE RECOGNITION LOOP
# -------------------------
def start_attendance_system():
    ensure_directories()

    # Load or create encodings
    data = load_encodings()

    if data is None:
        print("[INFO] No encoding found. Creating new encodings...")
        images, names = load_images()
        encs = encode_faces(images)
        save_encodings(encs, names)
        data = {"encodings": encs, "names": names}

    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    print("[INFO] Starting Smart Attendance System")
    print("[INFO] Press Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, faces)

        for enc, loc in zip(encs, faces):
            matches = face_recognition.compare_faces(known_encodings, enc)
            dist = face_recognition.face_distance(known_encodings, enc)

            name = "Unknown"
            if len(dist) > 0:
                idx = np.argmin(dist)
                if matches[idx]:
                    name = known_names[idx].upper()

            # Draw Rectangle
            y1, x2, y2, x1 = [v * 4 for v in loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------
# RUN SYSTEM
# -------------------------
if __name__ == "__main__":
    print("""
============ SMART ATTENDANCE SYSTEM ============

Options:
1. Start Attendance System
2. Enroll New Student
3. Exit
""")

    op = input("Choose option (1/2/3): ")

    if op == "1":
        start_attendance_system()
    elif op == "2":
        name = input("Enter student name: ").strip()
        enroll_from_webcam(name)
    else:
        print("Goodbye!")
