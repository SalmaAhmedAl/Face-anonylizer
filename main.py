import os
import cv2
import mediapipe as mp

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   #this step to save a new image
    
    
#read image
img = cv2.imread("./data/testImg.png")

H,W,_= img.shape


#detect faces
mp_face_detection = mp.solutions.face_detection
#create new object // model_selection (0 or 1)//
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    #print(out.detections)

#we want to extract this data  
if out.detections is not None: #try testNoFaceImg
    for detection in out.detections:
        location_data = detection.location_data
        bounding_box= location_data.relative_bounding_box 
        x1, y1, w, h = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height
        
        x1 =int(x1*W)
        y1 =int(y1*H)
        w =int(w*W)
        h =int(h*H)
        
       # img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,255,0), 10)

 # blur faces
img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (40, 40))
#cv2.imshow('img', img)
#cv2.waitKey(0)

#save image 
cv2.imwrite(os.path.join(output_dir, "output.jpg"), img)
