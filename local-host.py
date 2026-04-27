import cv2
import numpy as np
import math
import socket
import time

CAMERA_INDEX = 0
ROBOT_ID = 0
WAYPOINTS = [1, 2]
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ANGLE_THRESHOLD_DEG = 12
DIST_THRESHOLD_PX = 85
FRONT_TARGET_THRESHOLD_PX = 105
FORWARD_REACHED_THRESHOLD_PX = 120
LATERAL_REACHED_THRESHOLD_PX = 95
ROBOT_FRONT_OFFSET_PX = 95
EARLY_STOP_FRONT_PX = 85
EARLY_STOP_ANGLE_DEG = 20
REACHED_FRAMES_REQUIRED = 1
WAYPOINT_PAUSE = 0.8
LOST_TARGET_TIMEOUT = 1.0
LOST_TARGET_REACHED_DISTANCE = 125
TARGET_HOLD_TIMEOUT = 0.25

PI_IP = "192.168.137.211"
PI_PORT = 5000
SEND_INTERVAL = 0.20
RETRY_INTERVAL = 2.0


def marker_center(pts):
    return np.mean(pts, axis=0)


def robot_front_vector(pts):
    top_mid = (pts[0] + pts[1]) / 2.0
    center = np.mean(pts, axis=0)
    v = top_mid - center
    return v, center


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n


def angle_between_vectors_deg(v1, v2):
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    diff = math.degrees(a2 - a1)
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def send_command(sock, cmd):
    sock.sendall((cmd + "\n").encode("utf-8"))


def go_to_next_waypoint():
    global current_waypoint_index, mission_finished
    global last_seen_target_id, last_seen_target_time, last_seen_target_distance
    global last_sent_command, reached_counter

    reached_id = WAYPOINTS[current_waypoint_index]
    print("\n====================")
    print(f">>> WAYPOINT {reached_id} ATTEINT")
    if last_seen_target_distance is not None:
        print(f"Derniere distance vue: {last_seen_target_distance:.1f} px")
    print("====================")

    current_waypoint_index += 1
    last_sent_command = None
    reached_counter = 0
    last_seen_target_id = None
    last_seen_target_time = 0.0
    last_seen_target_distance = None

    if current_waypoint_index >= len(WAYPOINTS):
        mission_finished = True
        print(">>> MISSION TERMINEE")
        return True

    print(f">>> PASSAGE AU WAYPOINT {WAYPOINTS[current_waypoint_index]}")
    return False


aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Erreur ouverture camera {CAMERA_INDEX}")
    raise SystemExit

sock = None
last_connect_try = 0.0
last_sent_command = None
last_sent_time = 0.0
current_waypoint_index = 0
mission_finished = False
last_seen_target_id = None
last_seen_target_time = 0.0
last_seen_target_distance = None
reached_counter = 0
last_tracking_command = "STOP"
last_tracking_time = 0.0

print("Appuie sur q pour quitter")
print(f"Waypoints a suivre: {WAYPOINTS}")

try:
    while True:
        now = time.time()

        if sock is None and (now - last_connect_try) > RETRY_INTERVAL:
            last_connect_try = now
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((PI_IP, PI_PORT))
                print(f"Connecte au Pi {PI_IP}:{PI_PORT}")
            except Exception as e:
                print(f"Connexion impossible: {e}")
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None

        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture image")
            break

        frame = cv2.resize(frame, (960, 720))
        corners, ids, _ = detector.detectMarkers(frame)
        markers = {}

        if ids is not None:
            ids = ids.flatten()
            for i, corner in enumerate(corners):
                pts = corner.reshape((4, 2)).astype(np.float32)
                pts_int = pts.astype(int)
                center = marker_center(pts).astype(int)
                marker_id = int(ids[i])
                markers[marker_id] = pts

                for j in range(4):
                    cv2.line(frame, tuple(pts_int[j]), tuple(pts_int[(j + 1) % 4]), (0, 255, 0), 2)
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

                if marker_id == ROBOT_ID:
                    label = f"ROBOT {marker_id}"
                    color = (255, 0, 0)
                elif marker_id in WAYPOINTS:
                    label = f"WP {marker_id}"
                    color = (0, 255, 255)
                else:
                    label = f"ID {marker_id}"
                    color = (200, 200, 200)

                cv2.putText(frame, label, (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        command = "STOP"
        current_target_id = None

        if not mission_finished and ROBOT_ID in markers and current_waypoint_index < len(WAYPOINTS):
            robot_pts = markers[ROBOT_ID]
            current_target_id = WAYPOINTS[current_waypoint_index]

            if current_target_id in markers:
                target_pts = markers[current_target_id]
                robot_center = marker_center(robot_pts)
                target_center = marker_center(target_pts)
                front_vec, _ = robot_front_vector(robot_pts)
                heading = normalize(front_vec)
                front_point = robot_center + heading * ROBOT_FRONT_OFFSET_PX
                to_target_center = target_center - robot_center
                to_target_front = target_center - front_point
                distance = np.linalg.norm(to_target_center)
                front_distance = np.linalg.norm(to_target_front)
                angle_error = angle_between_vectors_deg(front_vec, to_target_center)
                forward_proj = float(np.dot(to_target_front, heading))
                lateral_error = float(abs(cross2d(heading, to_target_front)))

                last_seen_target_id = current_target_id
                last_seen_target_time = now
                last_seen_target_distance = min(distance, front_distance)

                print(
                    f"Approche WP {current_target_id} | "
                    f"d_center={distance:.1f} | "
                    f"d_front_real={front_distance:.1f} | "
                    f"forward={forward_proj:.1f} | "
                    f"lateral={lateral_error:.1f} | "
                    f"angle={angle_error:.1f}"
                )

                p1 = tuple(robot_center.astype(int))
                p2 = tuple((robot_center + heading * 120.0).astype(int))
                cv2.arrowedLine(frame, p1, p2, (255, 0, 255), 3)
                cv2.circle(frame, tuple(front_point.astype(int)), 7, (0, 165, 255), -1)
                cv2.line(frame, tuple(robot_center.astype(int)), tuple(target_center.astype(int)), (255, 255, 0), 2)
                cv2.line(frame, tuple(front_point.astype(int)), tuple(target_center.astype(int)), (0, 165, 255), 2)

                early_stop = (
                    front_distance < EARLY_STOP_FRONT_PX
                    and abs(angle_error) < EARLY_STOP_ANGLE_DEG
                    and forward_proj > -10
                )

                reach_center = distance < DIST_THRESHOLD_PX
                reach_front = front_distance < FRONT_TARGET_THRESHOLD_PX
                reach_pass = (
                    -10 < forward_proj < FORWARD_REACHED_THRESHOLD_PX
                    and lateral_error < LATERAL_REACHED_THRESHOLD_PX
                )
                reach_close_masking = (
                    front_distance < 115
                    and lateral_error < 130
                )

                reached_now = early_stop or reach_center or reach_front or reach_pass or reach_close_masking

                if reached_now:
                    reached_counter += 1
                    print(
                        f" -> validation WP {current_target_id} "
                        f"(early={early_stop}, center={reach_center}, front={reach_front}, "
                        f"pass={reach_pass}, close_mask={reach_close_masking}) "
                        f"[{reached_counter}/{REACHED_FRAMES_REQUIRED}]"
                    )
                else:
                    reached_counter = 0

                if reached_counter >= REACHED_FRAMES_REQUIRED:
                    command = "STOP"
                    if sock is not None:
                        try:
                            send_command(sock, "STOP")
                            send_command(sock, "STOP")
                            last_sent_command = "STOP"
                            last_sent_time = time.time()
                            print(">>> STOP anticipe envoye")
                        except Exception as e:
                            print(f"Erreur envoi STOP: {e}")
                            try:
                                sock.close()
                            except Exception:
                                pass
                            sock = None
                            last_sent_command = None
                    finished = go_to_next_waypoint()
                    if not finished:
                        time.sleep(WAYPOINT_PAUSE)
                else:
                    if angle_error > ANGLE_THRESHOLD_DEG:
                        command = "RIGHT"
                    elif angle_error < -ANGLE_THRESHOLD_DEG:
                        command = "LEFT"
                    else:
                        command = "FORWARD"
                    last_tracking_command = command
                    last_tracking_time = now

                cv2.putText(frame, f"Target ID: {current_target_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Angle: {angle_error:.1f} deg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"D center: {distance:.1f} px", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"D front real: {front_distance:.1f} px", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Forward: {forward_proj:.1f} px", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Lateral: {lateral_error:.1f} px", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Early stop: {early_stop}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                recently_seen_as_reached = (
                    last_seen_target_id == current_target_id
                    and (now - last_seen_target_time) < LOST_TARGET_TIMEOUT
                    and last_seen_target_distance is not None
                    and last_seen_target_distance < LOST_TARGET_REACHED_DISTANCE
                )

                if recently_seen_as_reached:
                    command = "STOP"
                    print(f"WP {current_target_id} perdu juste apres approche -> considere atteint")
                    if sock is not None:
                        try:
                            send_command(sock, "STOP")
                            send_command(sock, "STOP")
                            last_sent_command = "STOP"
                            last_sent_time = time.time()
                        except Exception as e:
                            print(f"Erreur envoi STOP: {e}")
                            try:
                                sock.close()
                            except Exception:
                                pass
                            sock = None
                            last_sent_command = None
                    finished = go_to_next_waypoint()
                    if not finished:
                        time.sleep(WAYPOINT_PAUSE)
                else:
                    if (now - last_tracking_time) < TARGET_HOLD_TIMEOUT:
                        command = last_tracking_command
                        print(f"WP {current_target_id} momentanement non detecte -> HOLD {command}")
                    else:
                        command = "STOP"
                        print(f"WP {current_target_id} non detecte -> STOP")
                    reached_counter = 0

                cv2.putText(frame, f"Target ID {current_target_id} non detecte", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        else:
            command = "STOP"
            reached_counter = 0

        if mission_finished:
            cv2.putText(frame, "Mission terminee", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        elif ROBOT_ID not in markers:
            print("ROBOT NON DETECTE")

        if sock is not None:
            try:
                if (command != last_sent_command) or ((now - last_sent_time) > SEND_INTERVAL):
                    send_command(sock, command)
                    print(f"Envoi: {command}")
                    last_sent_command = command
                    last_sent_time = now
            except Exception as e:
                print(f"Erreur envoi: {e}")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                last_sent_command = None

        cv2.putText(frame, f"Command: {command}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.putText(frame, f"Waypoint index: {current_waypoint_index}", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if current_waypoint_index < len(WAYPOINTS):
            cv2.putText(frame, f"Current target: {WAYPOINTS[current_waypoint_index]}", (20, 365),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Aruco Navigation - stop plus tot", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if sock is not None:
        try:
            send_command(sock, "STOP")
            sock.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()
