# importing the opencv
import cv2

# creating object of CascadeClassifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## Read image from directory
img = cv2.imread("photo.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# calling the CascadeClassifier object To geting Face dimensions
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# use for loop to create rectangle around the faces
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

resize = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

cv2.imshow("Detected", resize)
cv2.imwrite("face_detected.jpg", resize)
cv2.waitKey()
