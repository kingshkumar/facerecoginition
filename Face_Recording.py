import cv2
import numpy as np

# for face detection in video
capture=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier('FaceFront.xml')
Face_data=[]  # list to record cropped images ( faces)
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    ret,image = capture.read()
    print(ret,end='')
    if ret:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray)
        for x,y,w,h in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,0),5)
            myface=image[y:y+h,x:x+w,:]# cropping of image
            myface=cv2.resize(myface,(50,50))
            if len(Face_data)<=100:
                print(len(Face_data))
                Face_data.append(myface)
            cv2.putText(image,"Face Recording...",(x,y),font,1,(0,255,0),5)
        cv2.imshow('image',image)
        if cv2.waitKey(1) & 0xff==27 or len(Face_data)>100:
            break
capture.release()
cv2.destroyAllWindows()
print(Face_data[0:2])
arr=np.array(Face_data)
np.save('Kani.npy',arr)
face=Face_data[0]
import matplotlib.pyplot as plot
plot.imshow(face)
plot.show()
