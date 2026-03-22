import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    lowest_angle = 180.0 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        time_left = int(10 - elapsed)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if time_left <= 0:
                l_knee = calculate_angle(lm[23], lm[25], lm[27])
                r_knee = calculate_angle(lm[24], lm[26], lm[28])
                avg_knee = (l_knee + r_knee) / 2.0
                if avg_knee < lowest_angle: lowest_angle = avg_knee

                cv2.putText(frame, "CALIBRATING... SQUAT!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f"STARTING IN: {time_left}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Calibration', frame)
        if elapsed > 20 or (cv2.waitKey(1) & 0xFF == ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

    if lowest_angle < 170:
        with open("user_calibration.txt", "w") as f: f.write(str(int(lowest_angle + 10)))
        print(f"✅ Target saved: {int(lowest_angle + 10)}°")

if __name__ == "__main__":
    main()