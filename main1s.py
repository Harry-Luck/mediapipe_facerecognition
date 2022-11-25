import face_recognition as fr
import cv2
import numpy as np
import os
import pickle
import mediapipe

mp_face_detection = mediapipe.solutions.face_detection
face_detector =  mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence = 0.5)

path = "./train/"
url = 'rtsp://admin:@122.176.110.134:554/ch0_0.264'
known_names = []
images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

with open('mypickle.pickle' ,'rb') as f:
    loaded_obj = pickle.load(f)
known_name_encodings = loaded_obj
kk  =0
# image = "./test/face3.jpg"
# image = cv2.imread(test_image)
img12 = cv2.VideoCapture(url)
# img12 = cv2.VideoCapture('test6.mp4')

while (True):
    _, image = img12.read()
    # image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= cv2.resize(image, (650, 550))
    if kk< 1:
        results = face_detector.process(image)
        if results.detections:
            for face in results.detections:
                confidence = face.score
                bounding_box = face.location_data.relative_bounding_box
                x = int(bounding_box.xmin * image.shape[1]) - 20
                w = int(bounding_box.width * image.shape[1]) + 40
                y = int(bounding_box.ymin * image.shape[0]) -20
                h = int(bounding_box.height * image.shape[0]) +40

                image1 = image[y:y+h, x:x+w]
                face_locations = fr.face_locations(image1)
                face_encodings = fr.face_encodings(image1, face_locations)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = fr.compare_faces(known_name_encodings, face_encoding)
                    name = "unknown"
                    face_distances = fr.face_distance(known_name_encodings, face_encoding)
                    best_match = np.argmin(face_distances)

                    if matches[best_match]:
                        name = known_names[best_match]
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image, name, (x + 6, y - 6), font, 1.0, (255, 0, 255), 1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness = 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("1S", image)
    kk += 1
    if kk> 2:
        kk = 0
    # cv2.imwrite("./output.jpg", image)
    cv2.waitKey(2)
cv2.destroyAllWindows()
