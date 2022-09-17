import cv2,time

video=cv2.VideoCapture(0, cv2.CAP_DSHOW)
a=1

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True :

    a=a+1
    check,frame=video.read()
    #s=frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in faces :
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        #frame[x:x+w,y:y+h]=0
        #print(frame)
    
    cv2.imshow('video',frame)
    #cv2.imshow('videos',s)
    key=cv2.waitKey(1)
    if key==ord('q') :
        break

# print(check)
# print(frame)


video.release()
cv2.destroyAllWindows()