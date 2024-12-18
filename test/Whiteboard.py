import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a blank canvas
_, frame = cap.read()
canvas = np.zeros_like(frame)
previous_point = None
drawing_color = (255, 255, 255)  # White color for drawing

def is_only_index_raised(hand_landmarks, handedness):
    # Check if thumb is down
    thumb_is_down = False
    if handedness == "Right":
        thumb_is_down = hand_landmarks.landmark[4].x >= hand_landmarks.landmark[3].x
    else:
        thumb_is_down = hand_landmarks.landmark[4].x <= hand_landmarks.landmark[3].x
    
    # Check if index finger is up
    index_is_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    
    # Check if other fingers are down
    middle_is_down = hand_landmarks.landmark[12].y >= hand_landmarks.landmark[10].y
    ring_is_down = hand_landmarks.landmark[16].y >= hand_landmarks.landmark[14].y
    pinky_is_down = hand_landmarks.landmark[20].y >= hand_landmarks.landmark[18].y
    
    return thumb_is_down and index_is_up and middle_is_down and ring_is_down and pinky_is_down

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand label (Left/Right)
            label = hand_handedness.classification[0].label
            
            # Check if only index finger is raised
            index_only = is_only_index_raised(hand_landmarks, label)

            # Get index fingertip coordinates
            index_finger = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)

            # Draw only when index finger is raised
            if index_only:
                if previous_point is not None:
                    cv2.line(canvas, previous_point, (index_x, index_y), drawing_color, 2)
                previous_point = (index_x, index_y)
            else:
                previous_point = None

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine the canvas with the camera frame
    combined_image = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    
    # Display the combined image
    cv2.imshow('Whiteboard', combined_image)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()