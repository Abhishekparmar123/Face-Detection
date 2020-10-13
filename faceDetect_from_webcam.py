import cv2
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.VideoCapture(0)

time.sleep(2)

check, frame = image.read()
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

for x, y, w, h in face:
    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv2.imshow("face detected", frame)
cv2.waitKey()
cv2.destroyAllWindows()
