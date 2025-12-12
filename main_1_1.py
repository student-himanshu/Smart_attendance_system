# auto_face_attendance_with_cutoff.py
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import pickle
from datetime import datetime, date, time as dtime
import time

TRAIN_DIR = 'Training_images'
ENC_FILE = 'encodings.pkl'
CSV_FILE = 'record.csv'
CAM_INDEX = 0   # change to 1 if your webcam is at device 1

# Cutoff time for On-time vs Late (HH:MM:SS)
CUTOFF_TIME = dtime(hour=9, minute=30, second=0)  # default 09:30:00

os.makedirs(TRAIN_DIR, exist_ok=True)

def list_training_images():
    files = [f for f in os.listdir(TRAIN_DIR) if os.path.isfile(os.path.join(TRAIN_DIR, f))]
    return files

def load_images_and_names():
    images = []
    names = []
    files = list_training_images()
    for f in files:
        path = os.path.join(TRAIN_DIR, f)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {path}. Skipping.")
            continue
        images.append(img)
        names.append(os.path.splitext(f)[0])
    return images, names

def compute_encodings(images):
    enc_list = []
    for idx, img in enumerate(images):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if len(enc) == 0:
            print(f"[WARN] No face found in training image index {idx}.")
            continue
        enc_list.append(enc[0])
    return enc_list

def save_encodings(encodings, names):
    with open(ENC_FILE, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names, 'timestamp': time.time()}, f)
    print("[INFO] Encodings saved to", ENC_FILE)

def load_encodings():
    if not os.path.exists(ENC_FILE):
        return None
    with open(ENC_FILE, 'rb') as f:
        data = pickle.load(f)
    return data

def build_or_load_encodings(force_rebuild=False):
    if not force_rebuild:
        data = load_encodings()
        if data:
            imgs, names = load_images_and_names()
            if set(names) == set(data['names']):
                print("[INFO] Loaded encodings from cache.")
                return data['encodings'], data['names']
            else:
                print("[INFO] Training images changed. Rebuilding encodings.")
    imgs, names = load_images_and_names()
    encs = compute_encodings(imgs)
    valid_names = []
    for i, img in enumerate(imgs):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if len(enc) > 0:
            valid_names.append(names[i])
    save_encodings(encs, valid_names)
    return encs, valid_names

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
        df.to_csv(CSV_FILE, index=False)
        print("[INFO] Created", CSV_FILE)
    else:
        # ensure header contains Status
        df = pd.read_csv(CSV_FILE)
        if 'Status' not in df.columns:
            df['Status'] = ''
            df.to_csv(CSV_FILE, index=False)
            print("[INFO] Updated CSV header to include 'Status'")

# Attendance tracking for the current run; avoids repeated writes in one session
marked_today_runtime = set()

def get_today_entries(name):
    """Return a DataFrame of today's entries for `name` from CSV (may be empty)."""
    today_str = date.today().isoformat()
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        return pd.DataFrame(columns=['Name','Date','Time','Status'])
    if 'Date' not in df.columns:
        return pd.DataFrame(columns=['Name','Date','Time','Status'])
    entries = df[(df['Name'] == name) & (df['Date'] == today_str)]
    return entries

def determine_status(current_time):
    """Return 'On-time' if current_time <= CUTOFF_TIME else 'Late'."""
    t = dtime(hour=current_time.hour, minute=current_time.minute, second=current_time.second)
    return 'On-time' if t <= CUTOFF_TIME else 'Late'

def mark_attendance(name):
    """Mark attendance with Date, Time, Status. Enforce rules:
       - If person already has an On-time entry today -> do not mark again.
       - If person has only Late entries today and current mark is On-time -> allow (append).
       - Otherwise if any entry exists today and current status is not On-time -> do not append duplicate.
    """
    now = datetime.now()
    today_str = now.date().isoformat()
    time_str = now.strftime("%H:%M:%S")
    status = determine_status(now)

    # runtime quick-block to avoid many writes in same session
    if (name, today_str, status) in marked_today_runtime:
        return

    existing = get_today_entries(name)
    # if already has On-time today -> block
    if not existing.empty and 'On-time' in set(existing['Status'].astype(str)):
        # Already on-time today; do not mark again.
        print(f"[SKIP] {name} already marked On-time today. No new entry added.")
        return

    # if existing only Late and this mark is On-time -> allow append (correction)
    if not existing.empty:
        if 'Late' in set(existing['Status'].astype(str)) and status == 'On-time':
            # Append as correction
            df = pd.DataFrame([{'Name': name, 'Date': today_str, 'Time': time_str, 'Status': status}])
            df.to_csv(CSV_FILE, mode='a', index=False, header=False)
            marked_today_runtime.add((name, today_str, status))
            print(f"[MARKED-CORRECTED] {name} marked On-time at {time_str} (was Late earlier).")
            return
        else:
            # existing entries but no On-time; if current status is Late then block duplicate
            print(f"[SKIP] {name} already has an entry today ({existing[['Time','Status']].to_dict('records')}). No new entry added.")
            return

    # no existing entries today -> append normally
    df = pd.DataFrame([{'Name': name, 'Date': today_str, 'Time': time_str, 'Status': status}])
    df.to_csv(CSV_FILE, mode='a', index=False, header=False)
    marked_today_runtime.add((name, today_str, status))
    print(f"[MARKED] {name} at {time_str} -> {status}")

def enroll_person_from_webcam(cam, person_name, shots=5):
    person_safe = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    print(f"[ENROLL] Capturing {shots} photos for '{person_safe}'. Press SPACE to take each photo.")
    count = 0
    while count < shots:
        ret, frame = cam.read()
        if not ret:
            print("[WARN] Webcam read failed during enrollment.")
            break
        display = frame.copy()
        cv2.putText(display, f"Press SPACE to capture ({count}/{shots}) or 'c' to cancel", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Enroll", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            fname = os.path.join(TRAIN_DIR, f"{person_safe}_{count+1}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[ENROLL] Saved {fname}")
            count += 1
        elif key == ord('c'):
            print("[ENROLL] Cancelled by user.")
            break
    cv2.destroyWindow("Enroll")

def main():
    ensure_csv()
    encodings, names = build_or_load_encodings()
    print("[INFO] Known people:", names)

    cam = cv2.VideoCapture(CAM_INDEX)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam. Change CAM_INDEX if necessary.")

    print("[INFO] Press 'q' to quit, 'e' to enroll a new person (will open capture), 'r' to rebuild encodings.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[WARN] Frame capture failed, retrying...")
            time.sleep(0.1)
            continue

        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        for enc, loc in zip(face_encs, face_locs):
            matches = []
            face_dist = []
            if encodings:
                matches = face_recognition.compare_faces(encodings, enc)
                face_dist = face_recognition.face_distance(encodings, enc)
            name = "Unknown"
            if len(face_dist) > 0:
                best_idx = np.argmin(face_dist)
                if matches[best_idx]:
                    name = names[best_idx].upper()
            top, right, bottom, left = loc
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.rectangle(frame, (left, bottom-30), (right, bottom), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting.")
            break
        elif key == ord('e'):
            print("[INPUT] Enter name to enroll (in terminal): ", end='', flush=True)
            person_name = input().strip()
            if person_name:
                enroll_person_from_webcam(cam, person_name, shots=5)
                encodings, names = build_or_load_encodings(force_rebuild=True)
                print("[INFO] Rebuilt encodings after enrollment. Known:", names)
        elif key == ord('r'):
            print("[INFO] Rebuilding encodings now.")
            encodings, names = build_or_load_encodings(force_rebuild=True)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
