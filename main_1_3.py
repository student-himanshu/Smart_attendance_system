# smart_attendance_gui.py
"""
Smart Attendance System - Tkinter GUI
Features:
- Start/Stop attendance (webcam)
- Enroll new person (capture multiple images)
- Set cutoff time (On-time / Late)
- View today's attendance inside GUI
- Export attendance CSV
- Rebuild encodings
- Logging panel
"""

import os
import cv2
import time
import pickle
import threading
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime, date, time as dtime
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

# ---------------- CONFIG ----------------
TRAIN_DIR = "Training_images"
ENCODE_FILE = "encodings.pkl"
CSV_FILE = "attendance.csv"
CAM_INDEX = 0
DEFAULT_CUTOFF = dtime(9, 30, 0)
ENROLL_SHOTS = 5

os.makedirs(TRAIN_DIR, exist_ok=True)

# ---------------- Utilities & Backend ----------------
def ensure_csv():
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(CSV_FILE, index=False)

def load_images_and_names():
    imgs, names = [], []
    for f in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, f)
        img = cv2.imread(path)
        if img is not None:
            imgs.append(img)
            names.append(os.path.splitext(f)[0])
    return imgs, names

def compute_encodings(images):
    encs = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        e = face_recognition.face_encodings(rgb)
        if e:
            encs.append(e[0])
    return encs

def save_encodings(encodings, names):
    with open(ENCODE_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

def load_encodings():
    if not os.path.exists(ENCODE_FILE):
        return {"encodings": [], "names": []}
    with open(ENCODE_FILE, "rb") as f:
        return pickle.load(f)

def rebuild_encodings(log):
    log("Rebuilding encodings from training images...")
    imgs, names = load_images_and_names()
    encs = compute_encodings(imgs)
    save_encodings(encs, names)
    log(f"Encodings rebuilt: {len(encs)} faces known.")
    return encs, names

def determine_status(cutoff_time):
    now = datetime.now().time()
    return "On-Time" if now <= cutoff_time else "Late"

def append_attendance(name, status, log):
    ensure_csv()
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(CSV_FILE)
    # block if On-Time exists already for name today
    existing = df[(df['Name'] == name) & (df['Date'] == today)]
    if not existing.empty and ("On-Time" in existing['Status'].values):
        log(f"[SKIP] {name} already On-Time today.")
        return False
    # if existing Late and now On-Time -> append correction
    if not existing.empty and status == "On-Time" and ("Late" in existing['Status'].values):
        row = pd.DataFrame([{"Name": name, "Date": today, "Time": now, "Status": status}])
        row.to_csv(CSV_FILE, index=False, header=False, mode='a')
        log(f"[CORRECT] {name} marked On-Time at {now}.")
        return True
    if not existing.empty:
        log(f"[SKIP] {name} already has an entry today.")
        return False
    row = pd.DataFrame([{"Name": name, "Date": today, "Time": now, "Status": status}])
    row.to_csv(CSV_FILE, index=False, header=False, mode='a')
    log(f"[MARKED] {name} -> {status} at {now}")
    return True

# ---------------- GUI Application ----------------
class SmartAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Attendance System - GUI")
        self.geometry("900x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        ensure_csv()
        self.encodings_data = load_encodings()
        # If no encodings present, attempt build
        if not self.encodings_data.get("encodings"):
            rebuild_encodings(self.log)

        self.cutoff_time = DEFAULT_CUTOFF
        self.cam = None
        self.running = False
        self.thread = None
        self.marked_runtime = set()  # avoid duplicate writes in session

        self._build_ui()
        self.refresh_attendance_table()

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"[{ts}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def _build_ui(self):
        # Left controls frame
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Controls", font=("Segoe UI", 12, "bold")).pack(pady=(0,8))
        ttk.Button(left, text="Start Attendance", command=self.start).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Stop Attendance", command=self.stop).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Enroll New Person", command=self.enroll_dialog).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Rebuild Encodings", command=self.rebuild_enc_btn).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export CSV (Save As)", command=self.export_csv).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Refresh Attendance Table", command=self.refresh_attendance_table).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=6)

        ttk.Label(left, text="Cutoff Time (HH:MM)", font=("Segoe UI", 10)).pack()
        self.cutoff_entry = ttk.Entry(left)
        self.cutoff_entry.insert(0, self.cutoff_time.strftime("%H:%M"))
        self.cutoff_entry.pack(pady=4)
        ttk.Button(left, text="Set Cutoff", command=self.set_cutoff).pack(fill=tk.X)

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Status Log", font=("Segoe UI", 10)).pack(pady=(6,0))
        self.log_text = tk.Text(left, width=40, height=20, state='disabled')
        self.log_text.pack()

        # Right frame - camera frame + table
        right = ttk.Frame(self, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera display (uses OpenCV window for simplicity; still show status label)
        self.cam_label = ttk.Label(right, text="Camera: Not running", font=("Segoe UI", 10))
        self.cam_label.pack(anchor='w')

        # Attendance table
        ttk.Label(right, text="Today's Attendance", font=("Segoe UI", 12, "bold")).pack(pady=(8,0))
        cols = ("Name", "Date", "Time", "Status")
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

    def set_cutoff(self):
        txt = self.cutoff_entry.get().strip()
        try:
            h, m = txt.split(":")
            self.cutoff_time = dtime(int(h), int(m), 0)
            self.log(f"Cutoff set to {self.cutoff_time.strftime('%H:%M:%S')}")
        except Exception:
            messagebox.showerror("Invalid Time", "Enter cutoff as HH:MM (24-hour).")

    def enroll_dialog(self):
        name = simpledialog.askstring("Enroll", "Enter name of person:")
        if not name:
            return
        self.log(f"Starting enrollment for '{name}' (press Cancel in camera window to stop).")
        t = threading.Thread(target=self.enroll_from_camera, args=(name,), daemon=True)
        t.start()

    def enroll_from_camera(self, name):
        cam = cv2.VideoCapture(CAM_INDEX)
        count = 0
        safe = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        while count < ENROLL_SHOTS:
            ret, frame = cam.read()
            if not ret:
                self.log("Failed to read from camera during enrollment.")
                break
            display = frame.copy()
            cv2.putText(display, f"Press SPACE to capture ({count}/{ENROLL_SHOTS}) or 'c' to cancel", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Enroll - Press SPACE", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                fname = os.path.join(TRAIN_DIR, f"{safe}_{count+1}.jpg")
                cv2.imwrite(fname, frame)
                self.log(f"Saved enrollment image: {fname}")
                count += 1
            elif key == ord('c'):
                self.log("Enrollment cancelled by user.")
                break
        cam.release()
        cv2.destroyWindow("Enroll - Press SPACE")
        # rebuild encodings
        self.encodings_data = rebuild_encodings(self.log)
        self.log("Enrollment finished.")

    def rebuild_enc_btn(self):
        t = threading.Thread(target=lambda: setattr(self, 'encodings_data', rebuild_encodings(self.log)), daemon=True)
        t.start()

    def export_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if p:
            try:
                df = pd.read_csv(CSV_FILE)
                df.to_csv(p, index=False)
                self.log(f"Exported CSV to {p}")
                messagebox.showinfo("Exported", f"Saved CSV to:\n{p}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def refresh_attendance_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        df = pd.read_csv(CSV_FILE)
        today = date.today().isoformat()
        df = df[df['Date'] == today]
        for _, r in df.iterrows():
            self.tree.insert("", tk.END, values=(r['Name'], r['Date'], r['Time'], r['Status']))

    def start(self):
        if self.running:
            self.log("Attendance already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.log("Attendance started.")

    def stop(self):
        if not self.running:
            self.log("Attendance is not running.")
            return
        self.running = False
        # camera will be closed in loop
        self.log("Stopping attendance...")

    def _capture_loop(self):
        self.cam = cv2.VideoCapture(CAM_INDEX)
        if not self.cam.isOpened():
            self.log("Cannot open camera. Check CAM_INDEX.")
            self.running = False
            return
        self.cam_label.config(text="Camera: Running")
        while self.running:
            ret, frame = self.cam.read()
            if not ret:
                self.log("Frame read failed; retrying...")
                time.sleep(0.1)
                continue

            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locs = face_recognition.face_locations(rgb)
            face_encs = face_recognition.face_encodings(rgb, face_locs)
            known_encs = self.encodings_data.get("encodings", [])
            names = self.encodings_data.get("names", [])

            for enc, loc in zip(face_encs, face_locs):
                name = "Unknown"
                if known_encs:
                    matches = face_recognition.compare_faces(known_encs, enc)
                    dists = face_recognition.face_distance(known_encs, enc)
                    idx = np.argmin(dists) if len(dists)>0 else None
                    if idx is not None and matches[idx]:
                        name = names[idx].upper()
                # draw
                top, right, bottom, left = [v*4 for v in loc]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.rectangle(frame, (left, bottom-30), (right, bottom), (0,255,0), cv2.FILLED)
                status = determine_status(self.cutoff_time)
                cv2.putText(frame, f"{name} {status if name!='Unknown' else ''}", (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if name != "Unknown":
                    # attempt marking
                    key = (name, date.today().isoformat())
                    # We call append_attendance which handles duplicates & corrections
                    appended = append_attendance(name, status, self.log)
                    if appended:
                        # refresh UI table
                        self.refresh_attendance_table()

            cv2.imshow("Attendance - Press 'q' to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.log("Stop requested from OpenCV window.")
                self.running = False
                break

        # cleanup
        if self.cam:
            self.cam.release()
            self.cam = None
        cv2.destroyAllWindows()
        self.cam_label.config(text="Camera: Not running")
        self.log("Attendance stopped.")

    def on_close(self):
        if self.running:
            if not messagebox.askyesno("Quit", "Attendance is running. Stop and quit?"):
                return
            self.stop()
            time.sleep(0.5)
        self.destroy()

if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()
