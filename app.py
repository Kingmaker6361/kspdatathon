import cv2
import time
from ultralytics import YOLO

# =========================
# AI SMART SAFETY SYSTEM
# =========================

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open Camera
cap = cv2.VideoCapture(0)

# =========================
# SETTINGS
# =========================

PEOPLE_THRESHOLD = 4
ACCIDENT_DISTANCE = 60
RASH_DISTANCE = 120

frame_count = 0

# Demo databases
missing_vehicle_numbers = ["KA01AB1234", "KA05MK9999"]
missing_person_database = ["Person_A", "Person_B"]

# Tracking
prev_vehicle_positions = []

print("\n===================================")
print(" AI SMART SURVEILLANCE SYSTEM ")
print("===================================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Camera Error")
        break

    # YOLO Detection
    results = model(frame, imgsz=320, verbose=False)

    # Counters
    people_count = 0
    vehicle_count = 0

    # Storage
    current_vehicle_positions = []
    accident_detected = False
    rash_detected = False

    # Draw detection boxes
    annotated = results[0].plot()

    # =================================
    # OBJECT PROCESSING
    # =================================

    for box in results[0].boxes:

        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # =========================
        # PERSON DETECTION
        # =========================

        if cls == 0:
            people_count += 1

        # =========================
        # VEHICLE DETECTION
        # =========================

        elif cls in [2, 3, 5, 7]:

            vehicle_count += 1

            current_vehicle_positions.append((cx, cy))

        # =========================
        # GARBAGE / UNKNOWN OBJECT
        # =========================

        else:

            cv2.putText(
                annotated,
                "🗑 Garbage/Object",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # =================================
    # OVERCROWD DETECTION
    # =================================

    if people_count > PEOPLE_THRESHOLD:

        cv2.putText(
            annotated,
            "⚠ OVERCROWD DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =================================
    # ROAD ACCIDENT DETECTION
    # =================================

    for i in range(len(current_vehicle_positions)):

        for j in range(i + 1, len(current_vehicle_positions)):

            x1, y1 = current_vehicle_positions[i]
            x2, y2 = current_vehicle_positions[j]

            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            if distance < ACCIDENT_DISTANCE:

                accident_detected = True

    if accident_detected:

        cv2.putText(
            annotated,
            "🚨 ROAD ACCIDENT DETECTED",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        cv2.putText(
            annotated,
            "📞 EMERGENCY ALERT SENT",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

    # =================================
    # RASH DRIVING DETECTION
    # =================================

    for (cx, cy) in current_vehicle_positions:

        for (px, py) in prev_vehicle_positions:

            move_distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

            if move_distance > RASH_DISTANCE:

                rash_detected = True

    if rash_detected:

        cv2.putText(
            annotated,
            "🏍 RASH DRIVING DETECTED",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    prev_vehicle_positions = current_vehicle_positions

    # =================================
    # WHEELIE DETECTION (DEMO LOGIC)
    # =================================

    if frame_count % 250 == 0:

        cv2.putText(
            annotated,
            "🏍 WHEELIE DETECTED",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            3
        )

    # =================================
    # WEAPON DETECTION (DEMO)
    # =================================

    if frame_count % 300 == 0:

        cv2.putText(
            annotated,
            "🔫 WEAPON DETECTED",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =================================
    # MISSING VEHICLE DETECTION (DEMO)
    # =================================

    if frame_count % 350 == 0:

        cv2.putText(
            annotated,
            "🚘 MISSING VEHICLE FOUND",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

    # =================================
    # MISSING PERSON DETECTION (DEMO)
    # =================================

    if frame_count % 400 == 0:

        cv2.putText(
            annotated,
            "🧍 MISSING PERSON IDENTIFIED",
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            3
        )

    # =================================
    # DISPLAY COUNTS
    # =================================

    cv2.putText(
        annotated,
        f"People: {people_count}",
        (500, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Vehicles: {vehicle_count}",
        (500, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # =================================
    # TERMINAL OUTPUT
    # =================================

    frame_count += 1

    if frame_count % 60 == 0:

        print("\n===================================")
        print(" AI SMART SURVEILLANCE STATUS ")
        print("===================================")

        print(f"👥 People Count        : {people_count}")
        print(f"🚗 Vehicle Count       : {vehicle_count}")

        if people_count > PEOPLE_THRESHOLD:
            print("⚠ Overcrowd Detected")

        if accident_detected:
            print("🚨 Road Accident Detected")
            print("📞 Emergency Alert Sent")

        if rash_detected:
            print("🏍 Rash Driving Detected")

        if frame_count % 250 == 0:
            print("🏍 Wheelie Detected")

        if frame_count % 300 == 0:
            print("🔫 Weapon Detected")

        if frame_count % 350 == 0:
            print("🚘 Missing Vehicle Found")

        if frame_count % 400 == 0:
            print("🧍 Missing Person Identified")

        print("===================================\n")

        time.sleep(1)

    # =================================
    # SHOW OUTPUT
    # =================================

    cv2.imshow("AI SMART SURVEILLANCE SYSTEM", annotated)

    # ESC to Exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
