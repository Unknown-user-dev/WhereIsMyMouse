import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import time

mp_owo_hands = mp.solutions.hands
owo_hands_module = mp_owo_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
uwu_mouse_controller = Controller()
uwu_sensitivity = 4500

uwu_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
               [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [0, 5], [5, 9], [9, 13],
               [13, 17], [0, 17]]

uwu_capture = cv2.VideoCapture(0)

owo_click = False
uwu_click = False
uwu_mouse_pos = uwu_mouse_controller.position
last_uwu_wrist_pos = (-1, -1)
movement_delay = 0.01

while True:
    ret, uwu_frame = uwu_capture.read()

    uwu_frame = cv2.flip(uwu_frame, 1)

    uwu_frame_rgb = cv2.cvtColor(uwu_frame, cv2.COLOR_BGR2RGB)

    uwu_results = owo_hands_module.process(uwu_frame_rgb)

    if uwu_results.multi_hand_landmarks:
        for uwu_hand_landmarks in uwu_results.multi_hand_landmarks:
            for uwu_landmark in uwu_hand_landmarks.landmark:
                x = int(uwu_landmark.x * uwu_frame.shape[1])
                y = int(uwu_landmark.y * uwu_frame.shape[0])
                cv2.circle(uwu_frame, (x, y), 5, (0, 0, 255), -1)

            for uwu_connection in uwu_connections:
                x_start = int(uwu_hand_landmarks.landmark[uwu_connection[0]].x * uwu_frame.shape[1])
                y_start = int(uwu_hand_landmarks.landmark[uwu_connection[0]].y * uwu_frame.shape[0])
                x_end = int(uwu_hand_landmarks.landmark[uwu_connection[1]].x * uwu_frame.shape[1])
                y_end = int(uwu_hand_landmarks.landmark[uwu_connection[1]].y * uwu_frame.shape[0])
                cv2.line(uwu_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        if uwu_results.multi_hand_landmarks[0].landmark[8].y < uwu_results.multi_hand_landmarks[0].landmark[7].y:
            if not owo_click:
                owo_click = True
                print("OwO Finger Click")
                uwu_mouse_controller.click(Button.left, 1)
        else:
            owo_click = False

        if uwu_results.multi_hand_landmarks[0].landmark[12].y < uwu_results.multi_hand_landmarks[0].landmark[11].y:
            if not uwu_click:
                uwu_click = True
                print("UwU Finger Click")
                uwu_mouse_controller.click(Button.right, 1)
        else:
            uwu_click = False

        if (last_uwu_wrist_pos[0] != -1 and last_uwu_wrist_pos[1] != -1 and uwu_results.multi_hand_landmarks[0].landmark[4].x < \
                uwu_results.multi_hand_landmarks[0].landmark[3].x):
            if uwu_results.multi_hand_landmarks[0].landmark[0].x != last_uwu_wrist_pos[0] or \
                    uwu_results.multi_hand_landmarks[0].landmark[0].y != last_uwu_wrist_pos[1]:
                delta_x = (uwu_results.multi_hand_landmarks[0].landmark[0].x - last_uwu_wrist_pos[0]) * uwu_sensitivity
                delta_y = (uwu_results.multi_hand_landmarks[0].landmark[0].y - last_uwu_wrist_pos[1]) * uwu_sensitivity

                uwu_mouse_controller.move(delta_x, -delta_y)
        last_uwu_wrist_pos = (uwu_results.multi_hand_landmarks[0].landmark[0].x, uwu_results.multi_hand_landmarks[0].landmark[0].y)
        time.sleep(movement_delay)
    cv2.imshow('Hand Detection', uwu_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

uwu_capture.release()
cv2.destroyAllWindows()
