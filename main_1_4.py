"""
smart_attendance_monthly_autostop.py

Smart Attendance System (monthly) with AUTO-QUIT ON FIRST MARK option.

Features:
- Webcam-based Entry & Exit marking (face recognition)
- Monthly CSV files (attendance_records/YYYY_MM_attendance.csv)
- Present requires BOTH entry and exit during session by default (configurable)
- AUTO_QUIT_ON_FIRST: stop session as soon as first student's attendance is marked
- Enroll students via webcam
- Rebuild encodings
- Defaulter list and absent-by-student reports
"""

import os
import cv2
import time
import pickle
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, date

# ---------------- CONFIG ----------------
TRAIN_DIR = "Training_images"
ENCODE_FILE = "encodings.pkl"
ATT_FOLDER = "attendance_records"
CAM_INDEX = 0
ENROLL_SHOTS = 5
DEFAULTER_THRESHOLD = 18   # present-days <= threshold => defaulter

# Presence policy: require exit to count as present? (True = require exit)
REQUIRE_EXIT_FOR_PRESENT = True

# NEW: Auto-quit session after the first student is marked (Entry recorded).
AUTO_QUIT_ON_FIRST = True

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(ATT_FOLDER, exist_ok=True)


# ---------------- Utilities ----------------
def roster_from_training_images():
    files = [f for f in os.listdir(TRAIN_DIR) if os.path.isfile(os.path.join(TRAIN_DIR, f))]
    names = sorted({os.path.splitext(f)[0] for f in files})
    return names

def attendance_file_for_month(year: int, month: int):
    filename = f"{year:04d}_{month:02d}_attendance.csv"
    return os.path.join(ATT_FOLDER, filename)

def ensure_month_file(year: int, month: int):
    p = attendance_file_for_month(year, month)
    if not os.path.exists(p):
        df = pd.DataFrame(columns=["Name","Date","Session","EntryTime","ExitTime","Status"])
        df.to_csv(p, index=False)
    return p

def load_encodings():
    if os.path.exists(ENCODE_FILE):
        with open(ENCODE_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(data):
    with open(ENCODE_FILE, "wb") as f:
        pickle.dump(data, f)

def build_encodings_from_images():
    images = []
    names = []
    for fn in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        names.append(os.path.splitext(fn)[0])
    encs = []
    good_names = []
    for rgb, name in zip(images, names):
        enc = face_recognition.face_encodings(rgb)
        if enc:
            encs.append(enc[0])
            good_names.append(name)
        else:
            print(f"[WARN] no face found in training image for '{name}'")
    data = {"encodings": encs, "names": good_names}
    save_encodings(data)
    return data

def enroll_student_via_webcam(name: str):
    safe = "".join(c for c in name if c.isalnum() or c in (" ","_","-")).strip().replace(" ", "_")
    cam = cv2.VideoCapture(CAM_INDEX)
    if not cam.isOpened():
        print("Cannot open webcam")
        return
    print(f"Enrolling '{safe}'. Press SPACE to capture a photo, 'c' to cancel.")
    count = 0
    while count < ENROLL_SHOTS:
        ret, frame = cam.read()
        if not ret:
            continue
        disp = frame.copy()
        cv2.putText(disp, f"Press SPACE to capture ({count}/{ENROLL_SHOTS}), 'c' to cancel", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Enroll - Press SPACE", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            fname = os.path.join(TRAIN_DIR, f"{safe}_{count+1}.jpg")
            cv2.imwrite(fname, frame)
            print("Saved", fname)
            count += 1
        elif key == ord('c'):
            print("Enrollment cancelled")
            break
    cam.release()
    cv2.destroyAllWindows()
    print("Rebuilding encodings...")
    build_encodings_from_images()
    print("Enrollment complete.")


# ---------------- Session logic (with AUTO-QUIT) ----------------
def start_session(session_name: str):
    """
    Starts webcam capture for a class session.
    - Creates/opens monthly CSV
    - Records EntryTime on first detection for a student
    - Updates ExitTime on subsequent detections for the same student
    - If AUTO_QUIT_ON_FIRST is True, session stops immediately after first EntryTime is recorded
    """
    # prepare encodings
    data = load_encodings()
    if not data["encodings"]:
        print("No encodings found. Building from TRAIN_DIR...")
        data = build_encodings_from_images()
    known_enc = data["encodings"]
    known_names = data["names"]

    roster = roster_from_training_images()
    if not roster:
        print("[WARN] No roster found in Training_images/. Please enroll students first.")
        return

    # open webcam
    cam = cv2.VideoCapture(CAM_INDEX)
    if not cam.isOpened():
        print("Cannot open webcam")
        return

    today = date.today()
    year, month = today.year, today.month
    csv_path = ensure_month_file(year, month)

    # Load existing month df
    df_month = pd.read_csv(csv_path)

    # session records in memory
    session_records = {}  # Name -> {"EntryTime": str, "ExitTime": str}

    print("Session started. Press 'q' in the camera window or type 'stop' in terminal to end session.")
    if AUTO_QUIT_ON_FIRST:
        print("AUTO_QUIT_ON_FIRST is ENABLED: session will stop after the first student EntryTime is recorded.")
    else:
        print("AUTO_QUIT_ON_FIRST is DISABLED: session continues until manual stop.")

    # helper to format time
    def now_str():
        return datetime.now().strftime("%H:%M:%S")

    # Use a console listener to allow typing 'stop' to stop session
    import threading
    stop_flag = {"stop": False}

    def console_listener():
        while True:
            cmd = input().strip().lower()
            if cmd == "stop":
                stop_flag["stop"] = True
                break

    t_console = threading.Thread(target=console_listener, daemon=True)
    t_console.start()

    first_marked = False  # whether first Entry has been recorded

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.1)
                continue

            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb)
            face_encs = face_recognition.face_encodings(rgb, face_locs)

            for enc, loc in zip(face_encs, face_locs):
                name = "Unknown"
                if known_enc:
                    dists = face_recognition.face_distance(known_enc, enc)
                    idx = np.argmin(dists) if len(dists)>0 else None
                    if idx is not None and face_recognition.compare_faces(known_enc, enc)[idx]:
                        name = known_names[idx]

                if name != "Unknown":
                    rec = session_records.get(name, {"EntryTime": "", "ExitTime": ""})
                    # If no entry yet, set entry
                    if not rec["EntryTime"]:
                        rec["EntryTime"] = now_str()
                        print(f"[ENTRY] {name} at {rec['EntryTime']}")
                        # set first_marked flag
                        if not first_marked:
                            first_marked = True
                    else:
                        # update exit (latest detection)
                        rec["ExitTime"] = now_str()
                        print(f"[EXIT-UPDATE] {name} exit updated to {rec['ExitTime']}")
                    session_records[name] = rec

                # draw on frame (scale loc)
                top, right, bottom, left = loc
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                label = name if name != "Unknown" else ""
                cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow(f"Session: {session_name} - Press q to stop (or type 'stop')", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stop requested via webcam key.")
                break
            if stop_flag["stop"]:
                print("Stop requested via console input.")
                break

            # If AUTO_QUIT_ON_FIRST enabled and first_marked True, break loop
            if AUTO_QUIT_ON_FIRST and first_marked:
                print("[AUTO-QUIT] First attendance marked - stopping session.")
                # Small pause to ensure any ongoing exit update finishes
                time.sleep(0.5)
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    # finalize: write session records into month file (avoid duplicates)
    today_str = today.isoformat()
    existing_today_session = df_month[(df_month["Date"] == today_str) & (df_month["Session"] == session_name)]
    existing_names = set(existing_today_session["Name"].tolist())

    rows_to_append = []
    for student in roster:
        rec = session_records.get(student, {"EntryTime": "", "ExitTime": ""})
        entry = rec.get("EntryTime", "")
        exit_t = rec.get("ExitTime", "")

        # determine status
        if REQUIRE_EXIT_FOR_PRESENT:
            status = "Present" if entry and exit_t else "Absent"
        else:
            status = "Present" if entry else "Absent"

        if student in existing_names:
            # update existing row(s)
            idxs = df_month[(df_month["Date"] == today_str) & (df_month["Session"] == session_name) & (df_month["Name"] == student)].index
            if len(idxs) > 0:
                i = idxs[0]
                df_month.at[i, "EntryTime"] = entry
                df_month.at[i, "ExitTime"] = exit_t
                df_month.at[i, "Status"] = status
        else:
            rows_to_append.append({"Name": student, "Date": today_str, "Session": session_name,
                                   "EntryTime": entry, "ExitTime": exit_t, "Status": status})

    if rows_to_append:
        df_month = pd.concat([df_month, pd.DataFrame(rows_to_append)], ignore_index=True)

    df_month.to_csv(csv_path, index=False)
    print(f"Session '{session_name}' finalized and saved to {csv_path}")
    print(f"Session summary: {len(session_records)} students detected during session.")
    absent_list = [r["Name"] for r in rows_to_append if r["Status"] == "Absent"]
    if absent_list:
        print("Absent students (for this session):", absent_list)
    else:
        print("No absentees (everyone present by policy).")


# ---------------- Reporting ----------------
def defaulter_list_for_month(year: int, month: int, threshold: int = DEFAULTER_THRESHOLD):
    p = ensure_month_file(year, month)
    df = pd.read_csv(p)
    df_present = df[df["Status"].astype(str).str.lower() == "present"]
    df_present_unique = df_present[["Name","Date"]].drop_duplicates()
    counts = df_present_unique.groupby("Name").size().to_dict()
    roster = roster_from_training_images()
    defaulters = []
    for student in roster:
        cnt = counts.get(student, 0)
        if cnt <= threshold:
            defaulters.append((student, cnt))
    defaulters.sort(key=lambda x:(x[1], x[0]))
    return defaulters

def absent_sessions_by_student_for_month(student_name: str, year: int, month: int):
    p = ensure_month_file(year, month)
    df = pd.read_csv(p)
    df_student = df[df["Name"] == student_name]
    df_abs = df_student[df_student["Status"].astype(str).str.lower() == "absent"]
    return list(df_abs[["Date","Session"]].itertuples(index=False, name=None))


# ---------------- CLI ----------------
def print_menu():
    print("""
Smart Attendance System - Menu
Commands:
 start       -> Start a class session (enter session name; run until you press q in camera window or type 'stop')
 enroll      -> Enroll a new student via webcam
 rebuild     -> Rebuild face encodings from images in TRAIN_DIR
 defaulters  -> Generate defaulter list for a month
 absent_report -> Get list of absent sessions for a student for a month
 exit        -> Quit
""")

def main_cli():
    print("Smart Attendance System (Monthly) ready.")
    build_encodings_prompt = input("Do you want to (re)build encodings now? (y/n) [y]: ").strip().lower() or "y"
    if build_encodings_prompt == "y":
        print("Building encodings from images...")
        build_encodings_from_images()
    while True:
        print_menu()
        cmd = input("Enter command: ").strip().lower()
        if cmd == "start":
            session = input("Session name (e.g., Math_09_00): ").strip()
            if not session:
                print("Session name required.")
                continue
            start_session(session)
        elif cmd == "enroll":
            name = input("Student name to enroll: ").strip()
            if name:
                enroll_student_via_webcam(name)
        elif cmd == "rebuild":
            build_encodings_from_images()
            print("Encodings rebuilt.")
        elif cmd == "defaulters":
            ym = input("Enter month (YYYY-MM) or leave empty for current month: ").strip()
            if not ym:
                t = date.today()
                y, m = t.year, t.month
            else:
                try:
                    y, m = map(int, ym.split("-"))
                except:
                    print("Invalid format, use YYYY-MM.")
                    continue
            defs = defaulter_list_for_month(y, m)
            print(f"Defaulters for {y:04d}-{m:02d} (present-days <= {DEFAULTER_THRESHOLD}):")
            if not defs:
                print("No defaulters.")
            else:
                for name, cnt in defs:
                    print(f" {name}: {cnt} present days")
                out = os.path.join(ATT_FOLDER, f"{y:04d}_{m:02d}_defaulters.csv")
                pd.DataFrame(defs, columns=["Name","PresentDays"]).to_csv(out, index=False)
                print("Saved defaulter list to", out)
        elif cmd == "absent_report":
            student = input("Student name: ").strip()
            ym = input("Month (YYYY-MM) or empty for current month: ").strip()
            if not ym:
                t = date.today()
                y, m = t.year, t.month
            else:
                try:
                    y, m = map(int, ym.split("-"))
                except:
                    print("Invalid format.")
                    continue
            absent_sessions = absent_sessions_by_student_for_month(student, y, m)
            if not absent_sessions:
                print(f"No absent sessions for {student} in {y:04d}-{m:02d}.")
            else:
                print(f"Absent sessions for {student} in {y:04d}-{m:02d}:")
                for d, s in absent_sessions:
                    print(f" - {d} : {s}")
        elif cmd == "exit":
            print("Goodbye.")
            break
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main_cli()
