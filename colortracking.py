#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import RPIservo
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

servo = RPIservo.ServoCtrl()
servo.moveInit()
servo.start()

SERVO1_BASE = 1
SERVO2_ARM = 2
SERVO3_FOREARM = 3
SERVO4_GRIP = 4

BASE_MIN = 0
BASE_MAX = 180
ARM_MIN = 0
ARM_MAX = 180
FOREARM_MIN = 0
FOREARM_MAX = 180
GRIP_MIN = 0
GRIP_MAX = 180

INIT_BASE = 20
INIT_ARM = 30
INIT_FOREARM = 20
INIT_GRIP = 0

TARGET_BASE = 20
TARGET_ARM = 85
TARGET_FOREARM = 85
TARGET_GRIP = 90

base_angle = INIT_BASE
arm_angle = INIT_ARM
forearm_angle = INIT_FOREARM
grip_angle = INIT_GRIP

SCAN_MIN = 0
SCAN_MAX = 180
SCAN_STEP = 2
SCAN_DELAY = 0.04
TRACK_STEP = 1
TRACK_DELAY = 0.03

IMG_W = 640
IMG_H = 480
CENTER_X = IMG_W // 2

CENTER_TOL = 35

HSV_LOWER = np.array([85, 60, 60], dtype=np.uint8)
HSV_UPPER = np.array([115, 255, 255], dtype=np.uint8)
MIN_AREA = 1500

RIGHT_IS_PLUS = False

mode = "SCAN"
last_move = 0
lost_count = 0
LOST_LIMIT = 20
scan_dir = 1
grasp_done = False
center_start = None
last_frame_jpeg = None
frame_lock = threading.Lock()
smoothed_cx = None
smoothed_area = None

def clamp(x, a, b):
    return max(a, min(b, x))

def set_servo_angle(servo_id, angle):
    angle = int(clamp(angle, 0, 180))
    servo.moveAngle(servo_id, angle)
    return angle

def set_base(angle):
    global base_angle
    base_angle = int(clamp(angle, BASE_MIN, BASE_MAX))
    set_servo_angle(SERVO1_BASE, base_angle)

def set_arm(angle):
    global arm_angle
    arm_angle = int(clamp(angle, ARM_MIN, ARM_MAX))
    set_servo_angle(SERVO2_ARM, arm_angle)

def set_forearm(angle):
    global forearm_angle
    forearm_angle = int(clamp(angle, FOREARM_MIN, FOREARM_MAX))
    set_servo_angle(SERVO3_FOREARM, forearm_angle)

def set_grip(angle):
    global grip_angle
    grip_angle = int(clamp(angle, GRIP_MIN, GRIP_MAX))
    set_servo_angle(SERVO4_GRIP, grip_angle)

def move_slow_base(target, step=2, delay=0.03):
    global base_angle
    target = int(clamp(target, BASE_MIN, BASE_MAX))
    while abs(base_angle - target) > step:
        if base_angle < target:
            set_base(base_angle + step)
        else:
            set_base(base_angle - step)
        time.sleep(delay)
    set_base(target)
    time.sleep(delay)

def move_slow_arm(target, step=2, delay=0.03):
    global arm_angle
    target = int(clamp(target, ARM_MIN, ARM_MAX))
    while abs(arm_angle - target) > step:
        if arm_angle < target:
            set_arm(arm_angle + step)
        else:
            set_arm(arm_angle - step)
        time.sleep(delay)
    set_arm(target)
    time.sleep(delay)

def move_slow_forearm(target, step=2, delay=0.03):
    global forearm_angle
    target = int(clamp(target, FOREARM_MIN, FOREARM_MAX))
    while abs(forearm_angle - target) > step:
        if forearm_angle < target:
            set_forearm(forearm_angle + step)
        else:
            set_forearm(forearm_angle - step)
        time.sleep(delay)
    set_forearm(target)
    time.sleep(delay)

def move_slow_grip(target, step=2, delay=0.03):
    global grip_angle
    target = int(clamp(target, GRIP_MIN, GRIP_MAX))
    while abs(grip_angle - target) > step:
        if grip_angle < target:
            set_grip(grip_angle + step)
        else:
            set_grip(grip_angle - step)
        time.sleep(delay)
    set_grip(target)
    time.sleep(delay)

def set_initial_pose():
    move_slow_base(INIT_BASE)
    move_slow_arm(INIT_ARM)
    move_slow_forearm(INIT_FOREARM)
    move_slow_grip(INIT_GRIP)

def grasp_object():
    global grasp_done, mode
    print("[ACTION] mode saisie")
    mode = "GRAB"
    move_slow_base(TARGET_BASE)
    move_slow_arm(TARGET_ARM)
    move_slow_forearm(TARGET_FOREARM)
    move_slow_grip(TARGET_GRIP)
    time.sleep(0.8)
    print("[ACTION] transport vers depot")
    move_slow_base(80)
    time.sleep(0.8)
    print("[ACTION] ouverture pince pour depot")
    move_slow_grip(INIT_GRIP)
    time.sleep(0.8)
    print("[ACTION] retour position initiale")
    move_slow_arm(INIT_ARM)
    move_slow_forearm(INIT_FOREARM)
    move_slow_base(INIT_BASE)
    grasp_done = True
    print("[DONE] objet depose")

def detect_blue(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, None, None, 0, None, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < MIN_AREA:
        return False, None, None, area, None, mask

    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)

    if M["m00"] == 0:
        return False, None, None, area, None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return True, cx, cy, area, (x, y, w, h), mask

def smooth_value(prev, new, alpha=0.25):
    if prev is None:
        return float(new)
    return alpha * float(new) + (1.0 - alpha) * float(prev)

def move_base_toward(err_x):
    if RIGHT_IS_PLUS:
        if err_x > 0:
            set_base(base_angle + TRACK_STEP)
            print(f"[MOVE RIGHT] err_x={int(err_x)} angle={base_angle}")
        else:
            set_base(base_angle - TRACK_STEP)
            print(f"[MOVE LEFT] err_x={int(err_x)} angle={base_angle}")
    else:
        if err_x > 0:
            set_base(base_angle - TRACK_STEP)
            print(f"[MOVE RIGHT - INVERSE] err_x={int(err_x)} angle={base_angle}")
        else:
            set_base(base_angle + TRACK_STEP)
            print(f"[MOVE LEFT - INVERSE] err_x={int(err_x)} angle={base_angle}")

def annotate_frame(frame, detected, cx, cy, area, bbox, centered_seconds):
    out = frame.copy()
    cv2.line(out, (CENTER_X, 0), (CENTER_X, IMG_H), (255, 0, 0), 1)
    cv2.rectangle(out, (CENTER_X - CENTER_TOL, 0), (CENTER_X + CENTER_TOL, IMG_H), (0, 255, 255), 1)
    cv2.putText(out, f"MODE: {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, f"S1={base_angle} S2={arm_angle} S3={forearm_angle} S4={grip_angle}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(out, f"TIMER: {centered_seconds:.1f}s", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    if detected and bbox is not None:
        x, y, w, h = bbox
        err_x = int(cx - CENTER_X)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(out, f"cx={int(cx)} err_x={err_x} area={int(area)}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if abs(err_x) <= CENTER_TOL:
            cv2.putText(out, "OBJET CENTRE", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(out, "SUIVI...", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    else:
        cv2.putText(out, "OBJET NON DETECTE", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    return out

def update_stream_frame(frame):
    global last_frame_jpeg
    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        with frame_lock:
            last_frame_jpeg = jpeg.tobytes()

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = b"""
<html>
<head><title>PiCar Camera</title></head>
<body style="margin:0;background:#111;color:#fff;font-family:Arial;text-align:center;">
<h2>PiCar Camera Stream</h2>
<img src="/stream.mjpg" style="max-width:95vw;border:3px solid #444;">
</body>
</html>
"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                with frame_lock:
                    frame = last_frame_jpeg
                if frame is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def start_web_server():
    server = ThreadedHTTPServer(("0.0.0.0", 8080), StreamHandler)
    print("[WEB] flux camera sur http://192.168.137.211:8080")
    server.serve_forever()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

set_initial_pose()
time.sleep(1)

web_thread = threading.Thread(target=start_web_server, daemon=True)
web_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detected, cx, cy, area, bbox, mask = detect_blue(frame)
        now = time.time()
        centered_seconds = 0.0

        if detected:
            smoothed_cx = smooth_value(smoothed_cx, cx, alpha=0.25)
            smoothed_area = smooth_value(smoothed_area, area, alpha=0.25)
            cx_used = smoothed_cx
            area_used = smoothed_area
        else:
            cx_used = None
            area_used = None
            smoothed_cx = None
            smoothed_area = None

        if grasp_done:
            annotated = annotate_frame(
                frame,
                detected,
                cx_used if cx_used is not None else 0,
                cy if cy is not None else 0,
                area_used if area_used is not None else 0,
                bbox,
                centered_seconds
            )
            update_stream_frame(annotated)
            time.sleep(0.03)
            continue

        if mode == "SCAN":
            center_start = None

            if detected:
                mode = "TRACK"
                lost_count = 0
                print(f"[TRACK] objet detecte cx={int(cx_used)} area={int(area_used)}")
            else:
                if now - last_move > SCAN_DELAY:
                    new_angle = base_angle + scan_dir * SCAN_STEP

                    if new_angle >= SCAN_MAX:
                        new_angle = SCAN_MAX
                        scan_dir = -1
                    elif new_angle <= SCAN_MIN:
                        new_angle = SCAN_MIN
                        scan_dir = 1

                    set_base(new_angle)
                    print(f"[SCAN] angle={base_angle}")
                    last_move = now

        elif mode == "TRACK":
            if not detected:
                center_start = None
                lost_count += 1
                print(f"[LOST] {lost_count}/{LOST_LIMIT}")
                if lost_count >= LOST_LIMIT:
                    mode = "SCAN"
            else:
                lost_count = 0
                err_x = cx_used - CENTER_X

                if abs(err_x) <= CENTER_TOL:
                    if center_start is None:
                        center_start = now
                        print("[CENTER LOCK] depart timer")
                    centered_seconds = now - center_start
                    print(f"[CENTERED] cx={int(cx_used)} err_x={int(err_x)} stable={centered_seconds:.2f}s")

                    if centered_seconds >= 3.0:
                        print("[GRAB] objet centre pendant 3 secondes")
                        grasp_object()
                else:
                    center_start = None
                    if now - last_move > TRACK_DELAY:
                        move_base_toward(err_x)
                        last_move = now

        if center_start is not None and detected and not grasp_done:
            centered_seconds = max(0.0, now - center_start)

        annotated = annotate_frame(
            frame,
            detected,
            cx_used if cx_used is not None else 0,
            cy if cy is not None else 0,
            area_used if area_used is not None else 0,
            bbox,
            centered_seconds
        )
        update_stream_frame(annotated)
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[STOP] arret demande")

finally:
    cap.release()
