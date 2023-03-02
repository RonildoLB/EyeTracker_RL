#Imports
import cv2
import numpy as np
import dlib
import math
from math import hypot

#Definições de funções
def onChange(x):
    pass

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


#Acesso à câmera do dispositivo
cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

#Criação de Barras de ajuste de valores
cv2.createTrackbar("contraste", 'image', 100, 200, onChange)
cv2.createTrackbar("brilho", 'image', 0, 200, onChange)
cv2.createTrackbar("esquerda", 'image', 52, 100, onChange)
cv2.createTrackbar("direita", 'image', 57, 100, onChange)
cv2.createTrackbar("olho", 'image', 1, 1, onChange)

#Definições de variáveis
font = cv2.FONT_HERSHEY_DUPLEX #Fonte utilizada para escrever na 
ratio_dist_prev = None
text = None
count_noChange = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector() #Detector de faces
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Importação do preditor dos pontos de referências (facial landmark)

#Criação de estrutura de dados de imagem
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_var = cv2.SimpleBlobDetector_create(detector_params)


#Função de Processamento de imagem (Limiarização)
def blob_process(img, threshold, detector_var):
    #gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    _, img = cv2.threshold(img, 28, 255, cv2.THRESH_BINARY)
    thresh = img
    img = cv2.erode(img, None, iterations=2)
    erode = img
    img = cv2.dilate(img, None, iterations=2)
    dilate = img
    img = cv2.medianBlur(img, 9)
    blur = img
    img_arr = np.hstack((thresh, erode, dilate, blur))
    cv2.imshow("Processamento", img_arr)
    keypoints = detector_var.detect(img)
    return keypoints


#Função de detecção de piscada
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


#Função de pseudo gaze tracking
def gaze_track(ratio, limit_right, limit_left):
    if ratio >= limit_right:
        return "Direita"
    elif ratio <= limit_left:
        return "Esquerda"
    elif ratio > limit_left and ratio < limit_right:
        return "Centro"


while True:
    _, frame = cap.read()
    frame=cv2.flip(frame,1)

    contrast = cv2.getTrackbarPos("contraste", 'image')
    brightness = cv2.getTrackbarPos("brilho", 'image')
    limit_left = cv2.getTrackbarPos("esquerda", 'image') / 100
    limit_right = cv2.getTrackbarPos("direita", 'image') / 100
    alt_eye = cv2.getTrackbarPos("olho", 'image')
    alpha=(contrast / 100)
    beta=brightness
    frame = cv2.convertScaleAbs(frame, alpha=(contrast / 100), beta=brightness)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Detecta piscada
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 4:
            cv2.putText(frame, "PISCANDO", (90, 100), font, 1.6, (255, 200, 0), 2)

		# Detecção do olho
        if alt_eye == 1:
            eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(43).x, landmarks.part(43).y), (landmarks.part(44).x, landmarks.part(44).y), (landmarks.part(45).x, landmarks.part(45).y), (landmarks.part(46).x, landmarks.part(46).y), (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        else:
            eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(37).x, landmarks.part(37).y), (landmarks.part(38).x, landmarks.part(38).y), (landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(40).x, landmarks.part(40).y), (landmarks.part(41).x, landmarks.part(41).y)], np.int32)


        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_not(gray, gray, mask=mask)

        expansion = 8
        min_x = np.min(eye_region[:, 0]- expansion)
        max_x = np.max(eye_region[:, 0]+ expansion)
        min_y = np.min(eye_region[:, 1]- expansion)
        max_y = np.max(eye_region[:, 1]+ expansion)

        cv2.rectangle(frame, (min_x-1, min_y-1), (max_x+1, max_y+1), (0, 255, 0), 1)

        gray_eye = eye[min_y: max_y, min_x: max_x]

        threshold = 28

        keypoints = blob_process(gray_eye, threshold, detector_var)

        dist = -1

        output = None

        for keypoint in keypoints:
            pupil_x = math.ceil(keypoint.pt[0])
            pupil_y = math.ceil(keypoint.pt[1])
            dist = pupil_x
            cv2.circle(frame, (min_x + pupil_x, min_y + pupil_y), 0, (0, 255, 255), -1)
            output = [min_x + pupil_x, min_y + pupil_y]

        if dist != -1:
            ratio = dist / (max_x - min_x)
            text_rec = gaze_track(ratio, limit_right, limit_left)

            if(ratio_dist_prev != text_rec):
                count_noChange = 0
            
            ratio_dist_prev = text_rec
            count_noChange += 1

            if count_noChange > 7:
                text = text_rec
            
            color = (0,255,0)

            print(text, output)
            if text == "Esquerda": color = (0,0,255)
            if text == "Direita": color = (255,0,0)
            cv2.putText(frame, text, (90, 60), font, 1.6, color, 2)

        
        gray_eye = cv2.drawKeypoints(gray_eye, keypoints, gray_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        gray_eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        cv2.imshow("Olho", gray_eye)
        cv2.imshow("Rastreamento", eye)

    cv2.imshow("Imagem", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()