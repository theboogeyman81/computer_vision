import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def distance(p1, p2):
    return math.dist(p1, p2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                h, w, c = frame.shape

                # Thumb tip = 4
                x1 = int(handLms.landmark[4].x * w)
                y1 = int(handLms.landmark[4].y * h)

                # Index finger tip = 8
                x2 = int(handLms.landmark[8].x * w)
                y2 = int(handLms.landmark[8].y * h)

                # Draw circles
                cv2.circle(frame, (x1, y1), 10, (0,255,0), -1)
                cv2.circle(frame, (x2, y2), 10, (0,255,0), -1)

                # Calculate distance
                d = distance((x1,y1), (x2,y2))

                # Show distance
                cv2.putText(frame, f"Distance: {int(d)}", (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # Threshold for pinch
                if d < 40:   # Adjust threshold if needed
                    cv2.putText(frame, "PINCH DETECTED!", (10,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    cv2.circle(frame, (x1, y1), 20, (0,0,255), 3)
                    cv2.circle(frame, (x2, y2), 20, (0,0,255), 3)

        cv2.imshow("Pinch Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
