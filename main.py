import os
import argparse

import cv2
import mediapipe as mp

#create function called process_image takes an image and face_detection object
def process_image(img, face_detection):
    H, W, _ = img.shape
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

            #img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,255,0), 10)
            # blur faces       
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (40, 40))  
    return img     

args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   #this step to save a new image


#detect faces
mp_face_detection = mp.solutions.face_detection
#create new object // model_selection (0 or 1)//
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)
       
        img = process_image(img, face_detection)
        #save image 
        cv2.imwrite(os.path.join(output_dir, "output.jpg"), img) 

    elif args.mode in ["video"]:
       
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        #save_video
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),cv2.VideoWriter_fourcc(*'MP4V'),25, (frame.shape[1], frame.shape[0]))
        while ret:
            frame = process_image(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()


        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = process_image(frame, face_detection)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    # Capture the video frame
    # by frame
    
  
    # Display the resulting frame
    
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    
  


