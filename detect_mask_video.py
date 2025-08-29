import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from imutils.video import VideoStream
import pygame
import time

face_cascade=cv2.CascadeClassifier("C:/Users/Administrator/Downloads/haarcascade_frontalface_default.xml")

# Loading the trained model
mask_detector=load_model("C:/Users/Administrator/Downloads/face_mask_detector_model.h5")

# Defining alert sound file path

alert_sound_path="C:/Users/Administrator/Downloads/alert-33762.mp3"

pygame.mixer.init()

if os.path.exists(alert_sound_path):
    alert_sound=pygame.mixer.Sound(alert_sound_path)
else:
    alert_sound=None
    print("[WARNING] Alert sound file not found. Continuing without sound.")

#Starting the video streaming
print("[INFO] Starting video stream...")
vs=VideoStream(src=0).start()
time.sleep(2.0)

# Initialize a flag to control alert sound playback
sound_playing = False

while True:
    frame=vs.read()
    
    if frame is None:
        break
    
    # Resizing the frame to smaller size for faster processing
    frame=cv2.resize(frame,(600,450))
    
    #Fliping the frame horizontally for a more natural mirror view
    frame=cv2.flip(frame,1)
    
    # Converting the frame to grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Detecting faces in the grayscale frame
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                        minNeighbors=5, 
                                        minSize=(60,60)
                                        )
    # Initialize a variable to check if any 'no mask' face is detected in the current frame
    no_mask_detected_in_frame= False
    
    # Lopping over the detected faces
    for (x,y,w,h) in faces:
        # Extracting the face ROI
        
        face_roi=frame[y:y+h, x:x+w]
        face_roi=cv2.resize(face_roi,(128,128))
        face_roi=cv2.cvtColor(face_roi,cv2.COLOR_BGR2RGB)
        face_roi=img_to_array(face_roi)
        face_roi=np.expand_dims(face_roi,axis=0)
        face_roi=face_roi/255.0
        
        # Predicting mask or no mask
        (mask, without_mask)=mask_detector.predict(face_roi)[0]
        
        # Determining the class label and color for bounding box
        label="Mask" if mask > without_mask else "No Mask"
        color=(0,255,0) if label=="Mask" else (0,0,255)
        
        # Adding a check for the alert sound
        if label=="No Mask":
            no_mask_detected_in_frame= True
            
        # Displaying the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x,y),(x+w, y+h), color,2)
        
    if no_mask_detected_in_frame:
        if not sound_playing and alert_sound:
            alert_sound.play()
            sound_playing = True
            print("[ALERT] No mask detected! Playing alert sound.")
    else:
        if sound_playing:
            pygame.mixer.stop()
            sound_playing = False
            print("[ALERT] All masks detected. Stopping alert sound.")
        
    # Showing the output frame
    cv2.imshow("Face Mask Detector with live alert system", frame)
    key=cv2.waitKey(1) & 0xFF
    
    # If the 'q' key is pressed, break from the loop
    if key==ord("q"):
        break
    
# Cleaning up
print("[INFO] Stopping video streaming...")
cv2.destroyAllWindows()
vs.stop()

