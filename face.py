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
img12 = cv2.VideoCapture(0)
# img12 = cv2.VideoCapture('test6.mp4')

while (True):
    _, image = img12.read()
    # image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= cv2.resize(image, (650, 550))
    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = "unknown"
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Main", image)
    # cv2.imwrite("./output.jpg", image)
    cv2.waitKey(2)
cv2.destroyAllWindows()
