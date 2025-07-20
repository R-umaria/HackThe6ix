import cv2
import mediapipe as mp
import numpy as np
import time

model_path = 'models/face_landmarker.task'

# Setting up MediaPipe face landmarker modules and options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuring the face landmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True, # was prev. false
    output_facial_transformation_matrixes=False,  # added
    num_faces=1  # added
)

# Thresholds and constants for EAR, MAR, and detection logic

# Eye Aspect Ratio (EAR)
# it’s a way to measure how “tall” the eye is relative to its “width”
# Open ~ 0.25 - 0.35
# Half Closed ~ 0.2
# Closed ~ < 0.2
EAR_THRESHOLD = 0.25
BLENDSHAPE_THRESHOLD = 0.5
MAR_THRESHOLD = 0.5  # Adjusted for mouth openness detection
BLINK_FRAMES = 2 # Frames required for a blink - avoid false positives
DROWSY_FRAMES = 15 # ~0.6 sec at 24 FPS
DROWSY_SCORE_LIMIT = 5

# Eye landmark indices (based on MediaPipe's 468 landmark model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Counters and flags for tracking blinks, drowsiness, and yawns
blink_counter = 0
frame_counter = 0
drowsy_score = 0
mouth_open_frames = 0
blink_in_progress = False
drowsy_displayed = False
yawn_detected = False

start_time = time.time()

# Function to draw lines connecting landmarks of the eye
def draw_eye_contour(frame, landmarks, eye_indices, color=(0, 255, 0), thickness=2):
    points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in eye_indices]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i+1], color, thickness)
    # Optionally connect last to first to close the contour
    cv2.line(frame, points[-1], points[0], color, thickness)

# Draws lines connecting mouth inner landmarks
def draw_mouth_contour(frame, landmarks, mouth_indices, color=(0, 255, 0), thickness=2):
    points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in mouth_indices]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i+1], color, thickness)
    cv2.line(frame, points[-1], points[0], color, thickness)

# Computes Euclidean distance between two points
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# Computes Eye Aspect Ratio (EAR) for blink detection
def compute_ear(eye_landmarks):
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Computes Mouth Aspect Ratio (MAR) for yawn detection
def compute_mar(mouth_landmarks):
    vertical = np.mean([
        euclidean_dist(mouth_landmarks[3], mouth_landmarks[9]),  # 178-324
        euclidean_dist(mouth_landmarks[4], mouth_landmarks[8]),  # 87-318
        euclidean_dist(mouth_landmarks[5], mouth_landmarks[7])   # 14-402
    ])
    horizontal = euclidean_dist(mouth_landmarks[0], mouth_landmarks[10])  # 78-308
    return vertical / horizontal

# Start Webcam Feed
cap = cv2.VideoCapture(0)
frame_index = 0

# Run face landmarker on webcam feed
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # Create MediaPipe image
        timestamp_ms = int((time.time() - start_time) * 1000)  # Estimate timestamp for video mode
        results = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)  # Detect face landmarks
        frame_index += 1

        if results.face_landmarks:
            landmarks = results.face_landmarks[0] # Get the first detected face landmarks

            left_eye_landmarks = [landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE]

            left_ear = compute_ear(left_eye_landmarks)
            right_ear = compute_ear(right_eye_landmarks)
            avg_ear = (left_ear + right_ear) / 2.0 # Average EAR

            mouth_landmarks = [landmarks[i] for i in MOUTH_INNER]
            mar = compute_mar(mouth_landmarks) # Compute MAR

            # Default: no blendshape scores
            left_blink_score = 0
            right_blink_score = 0
            
            # Read blendshape scores for blink detection
            if results.face_blendshapes:
                blendshapes = results.face_blendshapes[0]
                for cat in blendshapes:
                    if cat.category_name == 'eyeBlinkLeft':
                        left_blink_score = cat.score
                    elif cat.category_name == 'eyeBlinkRight':
                        right_blink_score = cat.score
            
            # Determine if eyes are closed using blendshape or EAR
            eyes_closed = (
                (left_blink_score > BLENDSHAPE_THRESHOLD and right_blink_score > BLENDSHAPE_THRESHOLD) or 
                (avg_ear < EAR_THRESHOLD)
            )

            # Determine if mouth is open based on MAR
            mouth_open = mar > MAR_THRESHOLD

            # If eyes closed, update blink/drowsy logic
            if eyes_closed:
                frame_counter += 1

                if frame_counter >= BLINK_FRAMES and not blink_in_progress:
                    blink_counter += 1
                    blink_in_progress = True
                    print(f"Blink detected! Total: {blink_counter}")

                if frame_counter >= DROWSY_FRAMES:
                    drowsy_score += 1
                    if drowsy_score >= DROWSY_SCORE_LIMIT and not drowsy_displayed:
                        print("Drowsiness detected!")
                        drowsy_displayed = True
                    elif drowsy_score < DROWSY_SCORE_LIMIT:
                        drowsy_displayed = False
            else:
                frame_counter = 0
                drowsy_score = max(0, drowsy_score - 1)
                blink_in_progress = False
            
            # If mouth open for enough frames, mark as yawn
            if mouth_open:
                mouth_open_frames += 1
                if mouth_open_frames > 10 and not yawn_detected:
                    print("Yawn detected!")
                    yawn_detected = True
            else:
                mouth_open_frames = 0
                yawn_detected = False


            # Draw eye landmarks as dots on the eyes
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw contours around the eyes and mouth
            draw_eye_contour(frame, landmarks, LEFT_EYE)
            draw_eye_contour(frame, landmarks, RIGHT_EYE)
            draw_mouth_contour(frame, landmarks, MOUTH_INNER)

            # Overlay blink count and aspect ratios on screen
            cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            if drowsy_score >= DROWSY_SCORE_LIMIT:
                cv2.putText(frame, "DROWSY!", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if yawn_detected:
                cv2.putText(frame, "YAWN!", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the webcam frame
        cv2.imshow("Blink & Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

# destructor
cap.release()
cv2.destroyAllWindows()
