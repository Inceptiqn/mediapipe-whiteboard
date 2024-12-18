import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, handedness):

    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky fingertips
    finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
    count = 0

    if handedness == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1
    else:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            count += 1

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    return count

with mp_hands.Hands(
        static_image_mode=False,  # Real-time detection
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = hand_handedness.classification[0].label
                finger_count = count_fingers(hand_landmarks, label)

                # Draw landmarks and display the count
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display text on the frame
                text = f"{finger_count} Fingers"
                coords = (
                int(hand_landmarks.landmark[0].x * frame.shape[1]),
                int(hand_landmarks.landmark[0].y * frame.shape[0]) - 20
                )
                
                cv2.putText(frame, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Recognition with Finger Count", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()