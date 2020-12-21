import cv2
import numpy as np
import os 
from gtts import gTTS
from pygame import NOEVENT
from pygame import mixer  # Load the popular external library

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0
language = 'en'
# Emotions related to ids: example ==> Anger: id=0,  etc
names = ['Anger', 'Happy', 'neutral', 'sad', 'surprise', 'None'] 

speech = None

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
count=0
mixer.init()
while cam.isOpened():
    ret, img =cam.read()
    # img = cv2.imread("dwayne.jpg")
    #img = cv2.flip(img, 0) # Flip vertically
    count+=1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
        )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        speech = gTTS(text=str(id),lang =language,slow=False)
        if count%150==0:
            count=0
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  


            speech.save('dwayne_1.mp3')
        
            if mixer.music.get_endevent()==NOEVENT:
                mixer.music.load('dwayne_1.mp3')
                mixer.music.play()
            # if mixer.music.get_endevent()==NOEVENT:
               # mixer.music.unload()

    cv2.imshow("dwayne_1.jpg",img) 
   
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        mixer.quit()
        break;


print("\n [INFO] Done detecting and Image is saved")
cam.release()
cv2.destroyAllWindows()