from tkinter import CENTER
from ultralytics import YOLO
import cv2
import math
import time

#test

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# timing init
start_time = 0
end_time = 0
elapsed_time = 0

# center position calculation init
cx = 0
cy = 0

# velocity init
vel_x = 0
vel_y = 0

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    results = model(img, stream=True, classes=[41])
    elapsed_time = end_time - start_time
    start_time = time.time()
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            
            # center position calculation
            cx = x1 + ((x2-x1)*0.5)
            cy = y1 + ((y2-y1)*0.5)
            cent_pnt = (round(cx),round(cy))
            cent_txt = (round(cx),round(cy)+10)
            
            # velocity calculation
            vel_x = (cx-prev_cx)/elapsed_time
            vel_y = (cy-prev_cy)/elapsed_time
            t_avg_vel_x = (vel_x + prev_vel_x)*0.5
                                         
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, cent_pnt, radius=3, color=(255, 0, 255), thickness=2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, str(round(t_avg_vel_x, 2))+" px/s", cent_txt, font, fontScale, color, thickness)
            

    cv2.imshow('Webcam', img)

    end_time = time.time()
    
    prev_cx = cx
    prev_cy = cy
    prev_vel_x = vel_x
    
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()