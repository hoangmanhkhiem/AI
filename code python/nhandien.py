import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        print(results.face_landmarks)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(img,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
        # mp_drawing.DrawingSpec(color=(255,250,250),thickness=2,circle_radius=4))
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow("All Model", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()