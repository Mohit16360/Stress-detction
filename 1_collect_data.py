import cv2
import os

# Create directories for stress levels
stress_levels = ["stressed", "neutral", "relaxed"]
for level in stress_levels:
    if not os.path.exists(f"dataset/{level}"):
        os.makedirs(f"dataset/{level}")

cap = cv2.VideoCapture(0)
count = 0
label = "stressed"  # Change this to collect other categories

while count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Collecting Data", frame)
    cv2.imwrite(f"dataset/{label}/{count}.jpg", frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
