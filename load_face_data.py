import cv2
import numpy as np
capture=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier('FaceFront.xml')
font=cv2.FONT_HERSHEY_COMPLEX
name1=np.load('Kanishk.npy',allow_pickle=True).reshape(101,50*50*3)
name2=np.load('Parth.npy', allow_pickle=True).reshape(101,50*50*3)
data=np.concatenate([name1,name2])
users={'0':'Kanishk','1':'Parth'}
labels=np.zeros((202,1))
labels[:101]=0.0
labels[100:]=1.0
distances=[]
def distance(x1,x2):
    return np.sqrt(sum((x2-x1)**2))
def knn(x,train,k=5):
    size=train.shape[0]
    for i in range(size):
        dist=distance(x,train[i])
        distances.append(dist)
    sortedIndex=np.argsort(distances)
    sortedIndex=sortedIndex[:5]
    nearestNeighbours=[]
    for index in sortedIndex:
        nearestNeighbours.append(labels[index])
    count=np.unique(nearestNeighbours,return_counts=True)
    label=count[0][np.argmax(count[1])]
    return int(label)

        
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
            lab=knn(myface.flatten(),data)
            userName=users[str(lab)]
            cv2.putText(image,userName,(x,y),font,1,(0,255,0),5)
        cv2.imshow('image',image)
        if cv2.waitKey(1) & 0xff==27:
            break
capture.release()
cv2.destroyAllWindows()
