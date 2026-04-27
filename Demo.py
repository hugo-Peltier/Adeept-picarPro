import cv2
import numpy as np
import math
import socket
import time
import json

CAMERA_INDEX = 0
ROBOT_ID = 0
WAYPOINTS = [1, 2, 3]
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ANGLE_THRESHOLD_DEG = 12
ROBOT_FRONT_OFFSET_PX = 55

REACHED_FRAMES_REQUIRED = 3
WAYPOINT_PAUSE = 0.8
TARGET_HOLD_TIMEOUT = 0.25

REACH_CENTER_PX = 40
REACH_FRONT_PX = 50
REACH_ALIGNED_FRONT_PX = 90
REACH_ALIGNED_ANGLE_DEG = 18
REACH_ALIGNED_LATERAL_PX = 75

FINAL_APPROACH_FRONT_PX = 100
FINAL_APPROACH_CENTER_PX = 110
FINAL_APPROACH_FORWARD_MIN = -20
FINAL_APPROACH_FORWARD_MAX = 130
FINAL_APPROACH_LATERAL_PX = 90
FINAL_APPROACH_ANGLE_DEG = 10

JUST_LOST_TARGET_TIMEOUT = 0.7
ROBOT_LOST_HOLD_TIME = 0.35

PI_IP = "192.168.137.211"
PI_PORT = 5000
SEND_INTERVAL = 0.20
RETRY_INTERVAL = 2.0

IMU_UDP_IP = "0.0.0.0"
IMU_UDP_PORT = 5005
IMU_ALPHA = 0.8
IMU_TIMEOUT = 1.0

pick_blue_sent = False

def marker_center(pts):
    return np.mean(pts, axis=0)

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n

def normalize_angle_rad(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def rad_to_deg(a):
    return math.degrees(a)

def cross2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])

def signed_angle_deg(v1, v2):
    v1n = normalize(v1)
    v2n = normalize(v2)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    det = cross2d(v1n, v2n)
    ang = math.degrees(math.atan2(det, dot))
    return ang

def send_command(sock, cmd):
    sock.sendall((cmd + "\n").encode("utf-8"))

def robot_front_vector_from_pts(pts):
    top_mid = (pts[0] + pts[1]) / 2.0
    center = np.mean(pts, axis=0)
    v = top_mid - center
    return v, center

def theta_from_heading(heading):
    return math.atan2(heading[1], heading[0])

def heading_from_theta(theta):
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

def go_to_next_waypoint():
    global current_waypoint_index, mission_finished
    global last_seen_target_id, last_seen_target_time, last_seen_target_distance
    global last_sent_command, reached_counter
    global target_was_close, last_good_angle_error
    global last_tracking_command, last_tracking_time

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

    target_was_close = False
    last_good_angle_error = None

    last_tracking_command = "STOP"
    last_tracking_time = 0.0

    if current_waypoint_index >= len(WAYPOINTS):
        mission_finished = True
        print(">>> MISSION TERMINEE")
        return reached_id, True

    print(f">>> PASSAGE AU WAYPOINT {WAYPOINTS[current_waypoint_index]}")
    return reached_id, False

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

imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
imu_sock.bind((IMU_UDP_IP, IMU_UDP_PORT))
imu_sock.setblocking(False)

theta_imu = 0.0
imu_initialized = False
last_imu_packet_time = None
last_imu_receive_walltime = 0.0

current_waypoint_index = 0
mission_finished = False

last_seen_target_id = None
last_seen_target_time = 0.0
last_seen_target_distance = None

reached_counter = 0
last_tracking_command = "STOP"
last_tracking_time = 0.0

target_was_close = False
last_good_angle_error = None

last_robot_seen_time = 0.0
last_robot_center = None
last_heading = None

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

        while True:
            try:
                data, _addr = imu_sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))

                packet_time = float(msg["t"])
                gz = float(msg["gz"])

                if last_imu_packet_time is None:
                    last_imu_packet_time = packet_time
                else:
                    dt_imu = packet_time - last_imu_packet_time
                    last_imu_packet_time = packet_time

                    if 0.0 < dt_imu < 0.2:
                        theta_imu = normalize_angle_rad(theta_imu + gz * dt_imu)

                last_imu_receive_walltime = now

            except BlockingIOError:
                break
            except Exception as e:
                print(f"Erreur lecture UDP IMU: {e}")
                break

        imu_alive = (now - last_imu_receive_walltime) < IMU_TIMEOUT

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
        angle_error = None
        distance = None
        front_distance = None
        lateral_error = None
        theta_cam = None
        theta_fused = None
        forward_proj = None

        if not mission_finished and ROBOT_ID in markers and current_waypoint_index < len(WAYPOINTS):
            robot_pts = markers[ROBOT_ID]
            current_target_id = WAYPOINTS[current_waypoint_index]

            front_vec_cam, robot_center = robot_front_vector_from_pts(robot_pts)
            heading_cam = normalize(front_vec_cam)
            theta_cam = theta_from_heading(heading_cam)

            if not imu_initialized:
                theta_imu = theta_cam
                imu_initialized = True

            if imu_alive and imu_initialized:
                err = normalize_angle_rad(theta_cam - theta_imu)
                theta_fused = normalize_angle_rad(theta_imu + (1.0 - IMU_ALPHA) * err)
                theta_imu = theta_fused
            else:
                theta_fused = theta_cam
                theta_imu = theta_cam

            heading = heading_from_theta(theta_fused)

            if np.dot(heading, heading_cam) < 0.0:
                heading = heading_cam
                theta_fused = theta_cam
                theta_imu = theta_cam

            front_point = robot_center + heading * ROBOT_FRONT_OFFSET_PX

            last_robot_seen_time = now
            last_robot_center = robot_center.copy()
            last_heading = heading.copy()

            cv2.circle(frame, tuple(robot_center.astype(int)), 6, (255, 0, 0), -1)
            cv2.circle(frame, tuple(front_point.astype(int)), 7, (0, 165, 255), -1)

            cv2.arrowedLine(frame,
                            tuple(robot_center.astype(int)),
                            tuple((robot_center + heading * 120).astype(int)),
                            (255, 0, 255), 3)

            cv2.arrowedLine(frame,
                            tuple(robot_center.astype(int)),
                            tuple((robot_center + heading_cam * 95).astype(int)),
                            (255, 255, 255), 2)

            if current_target_id in markers:
                target_pts = markers[current_target_id]
                target_center = marker_center(target_pts)

                to_target_center = target_center - robot_center
                to_target_dir = normalize(to_target_center)
                to_target_front = target_center - front_point

                distance = float(np.linalg.norm(to_target_center))
                front_distance = float(np.linalg.norm(to_target_front))
                angle_error = signed_angle_deg(heading, to_target_dir)

                forward_proj = float(np.dot(to_target_front, heading))
                lateral_error = float(abs(cross2d(heading, to_target_front)))

                last_seen_target_id = current_target_id
                last_seen_target_time = now
                last_seen_target_distance = min(distance, front_distance)
                last_good_angle_error = angle_error

                if (
                    front_distance < FINAL_APPROACH_FRONT_PX or
                    distance < FINAL_APPROACH_CENTER_PX or
                    (
                        FINAL_APPROACH_FORWARD_MIN < forward_proj < FINAL_APPROACH_FORWARD_MAX and
                        lateral_error < FINAL_APPROACH_LATERAL_PX and
                        abs(angle_error) < FINAL_APPROACH_ANGLE_DEG
                    )
                ):
                    target_was_close = True

                cv2.line(frame, tuple(robot_center.astype(int)),
                         tuple(target_center.astype(int)), (255, 255, 0), 2)

                cv2.line(frame, tuple(front_point.astype(int)),
                         tuple(target_center.astype(int)), (0, 165, 255), 2)

                cv2.arrowedLine(frame,
                                tuple(robot_center.astype(int)),
                                tuple((robot_center + to_target_dir * 110).astype(int)),
                                (0, 255, 0), 2)

                cv2.circle(frame, tuple(target_center.astype(int)), 6, (0, 255, 255), -1)

                reach_center = distance < REACH_CENTER_PX
                reach_front = front_distance < REACH_FRONT_PX
                reach_aligned_close = (
                    front_distance < REACH_ALIGNED_FRONT_PX and
                    abs(angle_error) < REACH_ALIGNED_ANGLE_DEG and
                    lateral_error < REACH_ALIGNED_LATERAL_PX
                )

                reached_now = reach_center or reach_front or reach_aligned_close

                if reached_now:
                    reached_counter += 1
                    print(
                        f"WP {current_target_id} proche | "
                        f"d={distance:.1f} | d_front={front_distance:.1f} | "
                        f"forward={forward_proj:.1f} | lat={lateral_error:.1f} | "
                        f"angle={angle_error:.1f} [{reached_counter}/{REACHED_FRAMES_REQUIRED}]"
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
                        except Exception as e:
                            print(f"Erreur envoi STOP: {e}")
                            try:
                                sock.close()
                            except Exception:
                                pass
                            sock = None
                            last_sent_command = None

                    reached_id, finished = go_to_next_waypoint()

                    if reached_id == 3 and (not pick_blue_sent) and sock is not None:
                        try:
                            send_command(sock, "STOP")
                            time.sleep(0.2)
                            send_command(sock, "PICK_BLUE")
                            print(">>> PICK_BLUE envoye au Pi")
                            pick_blue_sent = True
                            command = "STOP"
                            last_sent_command = "PICK_BLUE"
                            last_sent_time = time.time()
                        except Exception as e:
                            print(f"Erreur envoi PICK_BLUE: {e}")
                            try:
                                sock.close()
                            except Exception:
                                pass
                            sock = None
                            last_sent_command = None

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

            else:
                just_lost_target = (
                    last_seen_target_id == current_target_id and
                    (now - last_seen_target_time) < JUST_LOST_TARGET_TIMEOUT
                )

                if target_was_close and just_lost_target:
                    print(f">>> WP {current_target_id} VALIDE (disparu en approche finale)")
                    command = "STOP"

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

                    reached_id, finished = go_to_next_waypoint()

                    if reached_id == 3 and (not pick_blue_sent) and sock is not None:
                        try:
                            send_command(sock, "STOP")
                            time.sleep(0.2)
                            send_command(sock, "PICK_BLUE")
                            print(">>> PICK_BLUE envoye au Pi")
                            pick_blue_sent = True
                            command = "STOP"
                            last_sent_command = "PICK_BLUE"
                            last_sent_time = time.time()
                        except Exception as e:
                            print(f"Erreur envoi PICK_BLUE: {e}")
                            try:
                                sock.close()
                            except Exception:
                                pass
                            sock = None
                            last_sent_command = None

                    if not finished:
                        time.sleep(WAYPOINT_PAUSE)

                else:
                    if (now - last_tracking_time) < TARGET_HOLD_TIMEOUT:
                        command = last_tracking_command
                    else:
                        command = "STOP"

                reached_counter = 0

        elif (not mission_finished) and current_waypoint_index < len(WAYPOINTS):
            current_target_id = WAYPOINTS[current_waypoint_index]
            robot_recently_lost = (now - last_robot_seen_time) < ROBOT_LOST_HOLD_TIME

            if imu_alive and imu_initialized and robot_recently_lost and last_robot_center is not None:
                heading = heading_from_theta(theta_imu)

                if last_heading is not None and np.dot(heading, last_heading) < 0.0:
                    heading = last_heading.copy()

                robot_center = last_robot_center
                front_point = robot_center + heading * ROBOT_FRONT_OFFSET_PX

                cv2.circle(frame, tuple(robot_center.astype(int)), 6, (255, 0, 0), -1)
                cv2.circle(frame, tuple(front_point.astype(int)), 7, (0, 165, 255), -1)
                cv2.arrowedLine(frame,
                                tuple(robot_center.astype(int)),
                                tuple((robot_center + heading * 120).astype(int)),
                                (180, 0, 255), 3)

                if current_target_id in markers:
                    target_center = marker_center(markers[current_target_id])
                    to_target_center = target_center - robot_center
                    to_target_dir = normalize(to_target_center)
                    angle_error = signed_angle_deg(heading, to_target_dir)

                    cv2.line(frame, tuple(robot_center.astype(int)),
                             tuple(target_center.astype(int)), (255, 255, 0), 2)

                    if angle_error > ANGLE_THRESHOLD_DEG:
                        command = "RIGHT"
                    elif angle_error < -ANGLE_THRESHOLD_DEG:
                        command = "LEFT"
                    else:
                        command = "FORWARD"

                    last_tracking_command = command
                    last_tracking_time = now
                else:
                    if (now - last_tracking_time) < TARGET_HOLD_TIMEOUT:
                        command = last_tracking_command
                    else:
                        command = "STOP"
            else:
                command = "STOP"
                reached_counter = 0

        else:
            command = "STOP"
            reached_counter = 0

        if sock is not None and not pick_blue_sent:
            try:
                if (command != last_sent_command) or ((now - last_sent_time) > SEND_INTERVAL):
                    send_command(sock, command)
                    print(f"CMD: {command}")
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

        y = 40
        line_h = 35

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (460, 360), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"WP index: {current_waypoint_index}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += line_h

        if current_waypoint_index < len(WAYPOINTS):
            cv2.putText(frame, f"Target: {WAYPOINTS[current_waypoint_index]}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += line_h

        if angle_error is not None:
            cv2.putText(frame, f"Angle: {angle_error:.1f} deg", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += line_h

        if distance is not None:
            cv2.putText(frame, f"Distance: {distance:.1f} px", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += line_h

        if front_distance is not None:
            cv2.putText(frame, f"D front: {front_distance:.1f} px", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += line_h

        if lateral_error is not None:
            cv2.putText(frame, f"Lateral: {lateral_error:.1f} px", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += line_h

        if forward_proj is not None:
            cv2.putText(frame, f"Forward: {forward_proj:.1f} px", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += line_h

        if theta_cam is not None:
            cv2.putText(frame, f"Theta cam: {rad_to_deg(theta_cam):.1f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += line_h

        if theta_fused is not None:
            cv2.putText(frame, f"Theta fused: {rad_to_deg(theta_fused):.1f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += line_h

        cv2.putText(frame, f"IMU alive: {imu_alive}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += line_h

        cv2.putText(frame, f"Target close: {target_was_close}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += line_h

        cv2.putText(frame, f"PICK_BLUE sent: {pick_blue_sent}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += line_h

        if mission_finished:
            cv2.putText(frame, "MISSION TERMINEE", (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow("NAV + IMU", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if sock is not None:
        try:
            send_command(sock, "STOP")
            sock.close()
        except Exception:
            pass

    imu_sock.close()
    cap.release()
    cv2.destroyAllWindows()
