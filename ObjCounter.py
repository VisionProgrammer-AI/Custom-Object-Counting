from ultralytics import YOLO
import cv2
import numpy as np
import random
import math

# Load your trained YOLO model
# model = YOLO('runs/detect/train/weights/best.pt')
model = YOLO('V11/best.pt')

# Open the video file
input_video_path = "boxes-move-along-conveyor-belt.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to configure the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = "BestBoxCounting.avi"  # Output video file name
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Tracker parameters
tracks = {}  # key: track_id, value: {'center': (x, y), 'lost': counter, 'color': (B, G, R), 'counted': bool, 'started_left': bool}
next_track_id = 0
distance_threshold = 50  # maximum Euclidean distance to consider the same object
max_lost = 5  # maximum frames to keep a track if detection is lost
confidence_threshold = 0.8  # only track detections with >80% confidence

# Initialize counter for objects that cross the line from left to right
counter = 0
# Define the x-coordinate for the vertical line
line_x = 200

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    results = model(frame)
    result = results[0]  # Process the first (and only) result for the frame

    # List to hold the centers of all valid detections in the current frame
    detection_centers = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy.numpy()
        confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf.numpy()

        for idx, box in enumerate(boxes):
            # Only process detection if confidence is above threshold
            if confidences[idx] < confidence_threshold:
                continue

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detection_centers.append((center_x, center_y))

            # Optionally, draw the raw detection center (red)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({center_x},{center_y})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # --- Tracking Logic ---
    assigned_detections = set()  # Keep track of detection indices already assigned to a track

    # Update existing tracks with the closest detection center if within threshold
    for track_id, track in list(tracks.items()):
        best_match = None
        best_distance = float('inf')
        for i, center in enumerate(detection_centers):
            if i in assigned_detections:
                continue
            dist = euclidean_distance(track["center"], center)
            if dist < best_distance:
                best_distance = dist
                best_match = i
        if best_match is not None and best_distance < distance_threshold:
            # Update track with new center and reset lost counter
            tracks[track_id]["center"] = detection_centers[best_match]
            tracks[track_id]["lost"] = 0
            assigned_detections.add(best_match)
        else:
            # If no detection is close enough, increment lost counter
            tracks[track_id]["lost"] += 1
            # Remove track if it has been lost for too long
            if tracks[track_id]["lost"] > max_lost:
                del tracks[track_id]

    # Create new tracks for unassigned detections
    for i, center in enumerate(detection_centers):
        if i not in assigned_detections:
            # Determine if the object started on the left side of the line
            started_left = center[0] < line_x
            # Assign a random color for the new track and mark it as not yet counted
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            tracks[next_track_id] = {"center": center, "lost": 0, "color": color, "counted": False, "started_left": started_left}
            next_track_id += 1

    # Draw the vertical line on the frame
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 2)

    # Check if any tracked object that started on the left crosses the line to the right and update counter
    for track_id, track in tracks.items():
        center = track["center"]
        # Only count if the object started on the left and now its center is to the right of the line
        if track["started_left"] and not track["counted"] and center[0] > line_x:
            counter += 1
            track["counted"] = True

    # Draw tracking information (colored circles and IDs) on the frame
    for track_id, track in tracks.items():
        center = track["center"]
        color = track["color"]
        cv2.circle(frame, center, 5, color, -1)
        cv2.putText(frame, f"ID:{track_id}", (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the counter on the top right corner
    cv2.putText(frame, f"Count: {counter}", (frame_width - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame (press 'q' to quit early)
    cv2.imshow("YOLO Inference with Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
