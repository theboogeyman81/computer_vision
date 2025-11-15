import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# To store previous thumb position
previous_thumb_y = None

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Landmarks
                index_tip = handLms.landmark[8]
                index_base = handLms.landmark[5]
                thumb_tip = handLms.landmark[4]

                # Convert to screen coords
                h, w, _ = img.shape
                ix_tip_y = int(index_tip.y * h)
                ix_base_y = int(index_base.y * h)
                thumb_y = int(thumb_tip.y * h)

                # Draw hand
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # Gesture logic ============================
                if previous_thumb_y is not None:

                    # Thumb sliding DOWN along the index finger (increase y)
                    if thumb_y > previous_thumb_y + 5:
                        print("Volume Decreased")

                    # Thumb sliding UP the index finger (decrease y)
                    elif thumb_y < previous_thumb_y - 5:
                        print("Volume Increased")

                previous_thumb_y = thumb_y
                # ===========================================

        cv2.imshow("Hand Volume Control", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
