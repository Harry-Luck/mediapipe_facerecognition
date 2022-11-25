import cv2
import mediapipe


mp_face_detection = mediapipe.solutions.face_detection
face_detector =  mp_face_detection.FaceDetection( min_detection_confidence = 0.6)

cap = cv2.VideoCapture('test1.mp4')
# image = cv2.imread('1.jpg')
while True:
    _, image = cap.read()
    image = cv2.resize(image, (600, 600))
    results = face_detector.process(image)
    if results.detections:
        for face in results.detections:
            confidence = face.score
            bounding_box = face.location_data.relative_bounding_box
            
            x = int(bounding_box.xmin * image.shape[1])
            w = int(bounding_box.width * image.shape[1])
            y = int(bounding_box.ymin * image.shape[0])
            h = int(bounding_box.height * image.shape[0])
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness = 2)
    cv2.waitKey(0)
    cv2.imshow('result', image)