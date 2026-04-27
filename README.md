# AdeeptPiCarPro — Autonomous Navigation & Manipulation

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-ArUco-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What is this?

This project is a school robotics build on top of the Adeept PiCar-Pro platform. The idea was to make the robot navigate autonomously through a series of ArUco markers, handle whatever gets in its way, and finally pick up and move an object on its own once it reaches the end of its route.

A Logitech webcam sits above the arena connected to a host PC and handles navigation. The small camera mounted on the robot arm takes over when it's time to grab something.

---

<!-- Add a photo of the full setup here (overhead view + robot) -->
![Setup overview](docs/images/setup.jpg)

---

## How it works

The mission runs in three phases:

**Phase 1 /Navigation**
The overhead camera detects ArUco markers on the ground. The host PC calculates the angle and distance between the robot and the next waypoint, then sends simple commands (FORWARD, LEFT, RIGHT, STOP) to the Raspberry Pi over a TCP socket. The robot works through its list of waypoints one by one.

**Phase 2/Obstacle handling**
If the ultrasonic sensors pick up something blocking the path, the robotic arm deploys and pushes the object out of the way before navigation resumes.

**Phase 3 /Object manipulation**
Once all waypoints are reached, the robot switches to manipulation mode. The arm camera scans the area, forward kinematics computes the joint angles needed to reach the detected object, and the arm picks it up and places it at a new random position.

---

<!-- Add a short demo video or GIF here -->
![Demo](docs/images/demo.gif)

---

## Project structure

```
adeept-picarpro/
│
├── navigation/
│   └── aruco_nav.py          
│
├── obstacle/
│   └── ultrasonic_push.py    
│
├── manipulation/
│   └── arm_fk.py             
├── robot/
│   └── pi_server.py         
│
├── config.py                 
└── README.md
```

---

## Hardware

- Adeept PiCar-Pro (robot base + arm)
- Raspberry Pi 3B+ or 4
- Logitech HD Pro webcam (overhead, connected to host PC)
- Camera module mounted on the robotic arm
- HC-SR04 ultrasonic sensors
- Host PC (Windows or Linux) connected to the Pi over Wi-Fi

---

<!-- Add a labeled hardware photo here -->
![Hardware](docs/images/hardware.jpg)

---

## Setup

Clone the repo:

```bash
git clone https://github.com/your-username/adeept-picarpro.git
cd adeept-picarpro
```

Install dependencies on the host PC:

```bash
pip install opencv-contrib-python numpy
```

Install dependencies on the Raspberry Pi:

```bash
pip install RPi.GPIO
```

Edit config.py to match your environment:

```python
PI_IP       = "192.168.137.211"
PI_PORT     = 5000
CAMERA_INDEX = 0
ROBOT_ID    = 0
WAYPOINTS   = [1, 2]
```

Start the server on the Pi:

```bash
python robot/pi_server.py
```

Then run the navigation script on the host PC:

```bash
python navigation/aruco_nav.py
```

Press Q to stop at any time.

---

## ArUco markers

Print markers from the DICT_4X4_50 dictionary and place them flat on the ground where the overhead camera can see them.

| ID | Role           |
|----|----------------|
| 0  | Robot          |
| 1  | Waypoint 1     |
| 2  | Waypoint 2     |
| 3+ | Extra waypoints |

Good lighting and no occlusion make a big difference in detection reliability.

---

<!-- Add a photo of the markers laid out on the ground -->
![Markers](docs/images/markers.jpg)

---

## Navigation logic

The heading controller is intentionally simple. If the angle error to the target is above a threshold the robot turns right, below the negative threshold it turns left, otherwise it goes straight. Arrival is confirmed when several distance and alignment criteria are met simultaneously: center distance, front distance, lateral error, and forward projection all fall within defined bounds.

---

## Known limitations

- The heading controller is a 3-state switch, not a PID — it works but is not smooth
- The FK model assumes a fixed table height and a known object size
- Detection is sensitive to lighting conditions and marker print quality

---

## What could be improved

- Replace the heading controller with a proper PID
- Add inverse kinematics for more flexible arm positioning
- Load waypoints dynamically from a config file or a small GUI
- Add ROS 2 support

---

## License

MIT — see LICENSE for details.
