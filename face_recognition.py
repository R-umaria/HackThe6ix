import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True, # was prev. false
    output_facial_transformation_matrixes=False,  # added
    num_faces=1  # added
)

# Eye Aspect Ratio (EAR)
# it’s a way to measure how “tall” the eye is relative to its “width”
# Open ~ 0.25 - 0.35
# Half Closed ~ 0.2
# Closed ~ < 0.2
EAR_THRESHOLD = 0.25

BLINK_FRAMES = 2 # Frames required for a blink - avoid false positives
DROWSY_FRAMES = 15 # ~0.6 sec at 24 FPS

blink_counter = 0
frame_counter = 0
drowsy = False
blink_in_progress = False

# Eye landmark indices (based on MediaPipe's 468 landmark model) - CONFIRM
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Function to draw lines connecting landmarks of the eye
def draw_eye_contour(frame, landmarks, eye_indices, color=(0, 255, 0), thickness=1):
    points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in eye_indices]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i+1], color, thickness)
    # Optionally connect last to first to close the contour
    cv2.line(frame, points[-1], points[0], color, thickness)

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def compute_ear(eye_landmarks):
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Start Webcam Feed
cap = cv2.VideoCapture(0)

frame_index = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # results = landmarker.detect(mp_image)
        timestamp_ms = int(frame_index * (1000 / 24))  # assuming 24 FPS
        results = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        frame_index += 1

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            left_eye_landmarks = [landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE]

            left_ear = compute_ear(left_eye_landmarks)
            right_ear = compute_ear(right_eye_landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            # Default: no blendshape scores
            left_blink_score = 0
            right_blink_score = 0

            if results.face_blendshapes:
                blendshapes = results.face_blendshapes[0]
                for cat in blendshapes:
                    if cat.category_name == 'eyeBlinkLeft':
                        left_blink_score = cat.score
                    elif cat.category_name == 'eyeBlinkRight':
                        right_blink_score = cat.score
            BLENDSHAPE_THRESHOLD = 0.5
            eyes_closed_by_blendshape = (left_blink_score > BLENDSHAPE_THRESHOLD and right_blink_score > BLENDSHAPE_THRESHOLD)
            eyes_closed_by_ear = (left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD)

            eyes_closed = eyes_closed_by_blendshape or eyes_closed_by_ear

            if eyes_closed:
                frame_counter += 1

                if frame_counter >= BLINK_FRAMES and not blink_in_progress:
                    blink_counter += 1
                    blink_in_progress = True
                    print(f"Blink detected! Total: {blink_counter}")

                if frame_counter >= DROWSY_FRAMES and not drowsy:
                    drowsy = True
                    print("Drowsiness detected!")
            else:
                frame_counter = 0
                drowsy = False
                blink_in_progress = False

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            draw_eye_contour(frame, landmarks, LEFT_EYE)
            draw_eye_contour(frame, landmarks, RIGHT_EYE)

            # Overlay info
            cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            if drowsy:
                cv2.putText(frame, "DROWSY!", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Blink & Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

# destructor
cap.release()
cv2.destroyAllWindows()
