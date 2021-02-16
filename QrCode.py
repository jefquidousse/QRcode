import cv2
import numpy as np
from pyzbar.pyzbar import decode


# For reading an image {
# image = cv2.imread('QR-image.png')
# code = decode(image)
# print(code)
# }

# Id 0 = id for camera
cap = cv2.VideoCapture(0)
# Width ID = 3
cap.set(3, 640)
# Height Id = 4
cap.set(4, 480)

# Class names
classNames = ['person', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test']

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confidenceThreshold = 0.5
def findObjects(outputs, image):
    hT, wT, cT = image.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                x, y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidenceThreshold, 0.3)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(image, f'{x}, {y}, {int(confs[i]*100)}%', (int((x + x+w)/2), int((y + y+h)/2)), cv2.FONT_ITALIC, 0.6, (255, 0, 255), 2)

while True:
    success, image = cap.read()

    # Convert image to blob, network understands this type
    blob = cv2.dnn.blobFromImage(image, 1/255, (320,320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    # print(type(outputs))
    findObjects(outputs, image)

    # Recognize QR-codes
    for barcode in decode(image):
        myData = barcode.data.decode('utf-8')
        print(myData)
        if myData == 'Dispenser':
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (255, 0, 255), 5)
            pts2 = barcode.rect
            x, y, w, h = pts2[0], pts2[1], pts2[2], pts2[3]
            cv2.putText(image, myData, (int((x + x+w)/2), int((y + y+h)/2)), cv2.FONT_ITALIC, 0.9, (255, 0, 255), 2)

    cv2.imshow('Result', image)
    cv2.waitKey(1)
