from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import mediapipe as mp
import numpy as np
import io

model_path = 'models/face_landmarker.task'

# Setting up MediaPipe face landmarker modules and options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuring the face landmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
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
EYE_CLOSURE_MIN_FRAMES = 2 # Frames required for a closed eyes - avoid false positives
DROWSY_FRAMES = 15 # ~0.6 sec at 24 FPS
DROWSY_SCORE_LIMIT = 5

# Eye landmark indices (based on MediaPipe's 468 landmark model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Counters and flags for tracking closed eyes, drowsiness, and yawns
eye_closure_count = 0
frame_counter = 0
drowsy_score = 0.0
mouth_open_frames = 0
drowsiness_level = 0
eye_closure_in_progress = False
drowsy_displayed = False
yawn_detected = False


def get_drowsiness_tier_and_color(level):
    if level <= 5:
        return "None", (255, 255, 255)
    elif level <= 25:
        return "low", (0, 255, 0)
    elif level <= 50:
        return "medium low", (0, 255, 255)
    elif level <= 75:
        return "medium", (0, 165, 255)
    elif level <= 90:
        return "medium high", (0, 0, 255)
    else:
        return "high", (0, 0, 128)

# Function to draw lines connecting landmarks of the eye
def draw_eye_contour(frame, landmarks, eye_indices, color=(0, 255, 0), thickness=2):
    points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in eye_indices]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i+1], color, thickness)
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

# Computes Eye Aspect Ratio (EAR) for closed eye detection
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

app = FastAPI()

# Dummy function to "process" the image (no actual processing for now)
async def image_processing(frame: np.ndarray) -> np.ndarray:
    # For now, just return the frame as-is (you can add actual processing here later)
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # Create MediaPipe image

        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            # Draw eye and mouth contours
            left_eye_landmarks = [landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE]
            mouth_landmarks = [landmarks[i] for i in MOUTH_INNER]

            left_ear = compute_ear(left_eye_landmarks)
            right_ear = compute_ear(right_eye_landmarks)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = compute_mar(mouth_landmarks)

            left_eye_score = 0
            right_eye_score = 0

            if results.face_blendshapes:
                blendshapes = results.face_blendshapes[0]
                for cat in blendshapes:
                    if cat.category_name == 'eyeBlinkLeft':
                        left_eye_score = cat.score
                    elif cat.category_name == 'eyeBlinkRight':
                        right_eye_score = cat.score

            eyes_closed = (
                (left_eye_score > BLENDSHAPE_THRESHOLD and right_eye_score > BLENDSHAPE_THRESHOLD) or
                (avg_ear < EAR_THRESHOLD)
            )

            mouth_open = mar > MAR_THRESHOLD

            if eyes_closed:
                global frame_counter, eye_closure_count, eye_closure_in_progress, drowsy_score, drowsiness_level
                frame_counter += 1

                if frame_counter >= EYE_CLOSURE_MIN_FRAMES and not eye_closure_in_progress:
                    eye_closure_count += 1
                    eye_closure_in_progress = True
                    print(f"Eye closure detected! Total: {eye_closure_count}")

                if frame_counter >= DROWSY_FRAMES:
                    drowsy_score = min(DROWSY_SCORE_LIMIT, drowsy_score + 0.1)
                else:
                    drowsy_score = max(0.0, drowsy_score - 0.05)
                
                drowsiness_level = int((drowsy_score / DROWSY_SCORE_LIMIT) * 100)
            else:
                frame_counter = 0
                drowsy_score = max(0.0, drowsy_score - 0.05)
                eye_closure_in_progress = False
            
            if mouth_open:
                global mouth_open_frames, yawn_detected
                mouth_open_frames += 1
                if mouth_open_frames > 10 and not yawn_detected:
                    yawn_detected = True
            else:
                mouth_open_frames = 0
                yawn_detected = False

            # Draw eye landmarks as dots on the eyes
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            draw_eye_contour(frame, landmarks, LEFT_EYE)
            draw_eye_contour(frame, landmarks, RIGHT_EYE)
            draw_mouth_contour(frame, landmarks, MOUTH_INNER)

            # comment these if in prod
            cv2.putText(frame, f"Closed Eyes: {eye_closure_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            tier, color = get_drowsiness_tier_and_color(drowsiness_level)
            cv2.putText(frame, f"Drowsiness: {tier} ({drowsiness_level}%)", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            if yawn_detected:
                cv2.putText(frame, "YAWN!", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Read the image data
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            # Invalid image but respond gracefully
            return JSONResponse({
                "success": False,
                "message": "Invalid image data",
            })

        processed_frame = await image_processing(frame)

        # Convert processed frame back to bytes for response
        _, img_encoded = cv2.imencode('.jpg', processed_frame)
        img_bytes = img_encoded.tobytes()

        # Capture end time after processing the image
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

    except Exception as e:
        # Log the error internally (to console here)
        print("Error processing image upload:", e)

        # Respond with success=False but never error code
        return JSONResponse({
            "success": False,
            "message": "Error processing image",
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
