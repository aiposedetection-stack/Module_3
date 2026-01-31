import cv2
import mediapipe as mp
import csv
import time
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
OUTPUT_CSV = "pose_data.csv"
IMAGE_FOLDER = "captured_frames"

# Create the folder if it doesn't exist
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Camera
cap = cv2.VideoCapture(0)

# ==========================================
# 📝 PREPARE CSV HEADER
# ==========================================
header = ["frame_id", "timestamp"]
for i in range(33):
    header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

print(f"✅ Recording started!")
print(f"   - Images saved to: {IMAGE_FOLDER}/")
print(f"   - Data saved to: {OUTPUT_CSV}")
print("Press 'q' to stop.")

# ==========================================
# 🎬 MAIN LOOP
# ==========================================
start_time = time.time()
frame_count = 0

while cap.isOpened():
    # 1. ACQUIRE FRAME
    ret, frame = cap.read()
    if not ret: break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # 2. SAVE RAW IMAGE FRAME (Before drawing lines on it)
    # We construct a filename like: captured_frames/frame_0.jpg
    image_filename = os.path.join(IMAGE_FOLDER, f"frame_{frame_count}.jpg")
    cv2.imwrite(image_filename, frame)

    # 3. POSE ESTIMATION
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 4. SAVE DATA TO CSV
    if results.pose_landmarks:
        # Draw skeleton on screen (for your view only)
        # We draw on 'frame', but we ALREADY saved the clean image above.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        row = [frame_count, time.time() - start_time] # Add ID and Time
        
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        with open(OUTPUT_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # 5. UI UPDATES
    cv2.putText(frame, f"Saved: {frame_count}", (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Frame Capture', frame)
    
    frame_count += 1 # Increment counter
    
    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Process Complete. Saved {frame_count} frames.")