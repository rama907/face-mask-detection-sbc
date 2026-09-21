# Aplikasi pendeteksi pemakaian masker dan jarak aman (webcam, komputer papan tunggal).
# Dibangun di atas proyek open source Face-Mask-Detection oleh chandrikadeb7 (MIT):
# https://github.com/chandrikadeb7/Face-Mask-Detection
# Jalankan dari folder root repositori ini: python app.py
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from math import pow, sqrt
from multiprocessing import Process
import numpy as np
import argparse
import imutils
import time
import cv2
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Nilai Konstan
camera = 0
preprocessing = False
calculateConstant_x = 300
calculateConstant_y = 615
personLabelID = 15.00
debug = True
accuracyThreshold = 0.4
RED = (0,0,255)
YELLOW = (0,255,255)
GREEN = (0,255,0)
write_video = False

#Pilih Audio untuk notif harap pakai masker & harap jaga jarak
def play():
    os.system('mpg321 "%s" &' % os.path.join(BASE_DIR, 'audio', 'harap_pakai_masker.mp3')) #pakai mp3
    #os.system('cvlc harap_pakai_masker.wav &') #kalo pakai format wav

def play2():
    os.system('mpg321 "%s" &' % os.path.join(BASE_DIR, 'audio', 'harap_jaga_jarak.mp3')) #pakai mp3
    
def detect_and_predict_mask(frame, faceNet, maskNet):
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)
# I used CLAHE preprocessing algorithm for detect humans better.
# HSV (Hue, Saturation, and Value channel). CLAHE uses value channel.
# Value channel refers to the lightness or darkness of a colour. An image without hue or saturation is a grayscale image.
def CLAHE(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def centroid(startX,endX,startY,endY):
    centroid_x = round((startX+endX)/2,4)
    centroid_y = round((startY+endY)/2,4)
    bboxHeight = round(endY-startY,4)
    return centroid_x,centroid_y,bboxHeight

def calcDistance(bboxHeight):
    distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
    return distance

def drawResult(frame,position):
    for i in position.keys():
        if i in highRisk:
            rectangleColor = RED
        else:
            rectangleColor = GREEN
        (startX, startY, endX, endY) = detectionCoordinates[i]

        cv2.rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)
        

if __name__== "__main__":

    caffeNetwork = cv2.dnn.readNetFromCaffe("./SSD_MobileNet_prototxt.txt", "./SSD_MobileNet.caffemodel")
    #for camera in glob.glob("/dev/video?"):
    cap = cv2.VideoCapture(camera)
    
    #Untuk memperkecil frame dari hasil objek pendeteksi (untuk mengurangi lag)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 350) 
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 350) 
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    prototxtPath = os.path.join(os.getcwd(), 'face_detector', 'deploy.prototxt')
    weightsPath = os.path.join(os.getcwd(), 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")


    while cap.isOpened():

        debug_frame, frame = cap.read()
        highRisk = set()
        mediumRisk = set()
        position = dict()
        detectionCoordinates = dict()

        if not debug_frame:
            print("Video tidak bisa dibuka!")
            break

        if preprocessing:
            frame = CLAHE(frame)

        (imageHeight, imageWidth) = frame.shape[:2]
        pDetection = cv2.dnn.blobFromImage(cv2.resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)

        caffeNetwork.setInput(pDetection)
        detections = caffeNetwork.forward()
        
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
        for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Menggunakan Masker" if mask > withoutMask else "Tidak menggunakan Masker"
            color = (0, 255, 0) if label == "Menggunakan Masker" else (0, 0, 255)
            
            #putar suara menggunakan perpustakaan audio
            #Notifikasi Harap Menggunakan Masker
            if label == "Tidak menggunakan Masker":
                P = Process(name="pakaimasker",target=play)
                P.start() # Inisialisasi Proses
                time.sleep(2)
            else:
                try:
                    P.terminate()
                except:
                    pass
                
            # include the probability in the label
            #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            

        for i in range(detections.shape[2]):

            accuracy = detections[0, 0, i, 2]
            if accuracy > accuracyThreshold:
                # Detection class and detection box coordinates.
                idOfClasses = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                (startX, startY, endX, endY) = box.astype('int')

                if idOfClasses == personLabelID:
                    # Default drawing bounding box.
                    bboxDefaultColor = (255,255,255)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bboxDefaultColor, 2)
                    detectionCoordinates[i] = (startX, startY, endX, endY)

                    # Centroid of bounding boxes
                    centroid_x, centroid_y, bboxHeight = centroid(startX,endX,startY,endY)                    
                    distance = calcDistance(bboxHeight)
                    # Centroid in centimeter distance
                    centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                    centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                    position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)

        #Penghitung resiko menggunkan posisi jarak
        for i in position.keys():
            for j in position.keys():
                if i < j:
                    distanceOfBboxes = sqrt(pow(position[i][0]-position[j][0],2) 
                                          + pow(position[i][1]-position[j][1],2) 
                                          + pow(position[i][2]-position[j][2],2)
                                          )
                    if distanceOfBboxes < 150: # 150cm atau lebih rendah
                        highRisk.add(i),highRisk.add(j)
                        #Notifikasi Harap Jaga Jarak
                        P = Process(name="jagajarak",target=play2)
                        P.start()
                        time.sleep(2)
                    elif distanceOfBboxes < 200 > 150: # antara 150 dan 200
                        mediumRisk.add(i),mediumRisk.add(j)
                    else:
                        try:
                            P.terminate()
                        except:
                            pass
            
       

        cv2.putText(frame, "Orang dengan resiko Tinggi : " + str(len(highRisk)) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Orang dengan resiko sedang : " + str(len(mediumRisk)) , (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "Orang yang terdeteksi : " + str(len(detectionCoordinates)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		
        drawResult(frame, position)
        if write_video:            
            output_movie.write(frame)
        cv2.imshow('frame', frame)
        waitkey = cv2.waitKey(1)
        if waitkey == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
		
