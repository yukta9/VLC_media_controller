import cv2
import numpy as np
import math
import subprocess
import pyautogui
import time
#pyautogui.PAUSE = 0.0
video=cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1100)
cv2.namedWindow('track')
#cv2.resizeWindow('track',600,600)


face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def hue(x) :
    pass 



cv2.createTrackbar('L-H','track',0,179,hue)
cv2.createTrackbar('L-s','track',65,255,hue)
cv2.createTrackbar('L-v','track',40,255,hue)
cv2.createTrackbar('h-H','track',21,179,hue)
cv2.createTrackbar('h-s','track',255,255,hue)
cv2.createTrackbar('h-v','track',255,255,hue)


j=1
last=0
pause=False
mute=False
isUP=False
def getHist():
    while True :
        
        ret,frame=video.read()
        frame=cv2.flip(frame,1)
        obj=frame[100:200,100:200]
        cv2.rectangle(frame,(100,100),(200,200),(0,255,0),3)
        if cv2.waitKey(1)==ord('a') :
            object_color=obj
            cv2.destroyWindow('frame')
            break
        cv2.imshow('frame',frame)
    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)

    return cv2.calcHist([object_color_hsv], [0, 1], None,
                                [10, 15], [0, 180, 0, 256])
#cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)   
object_hist = getHist()


def getFaces(gray,frame):
    faces=face.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=7)
    for x,y,w,h in faces :
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            frame[y:y+h+80,x:x+w]=0

def filterImg(roi) :
        lh=cv2.getTrackbarPos("L-H","track")
        ls=cv2.getTrackbarPos("L-s","track")
        lv=cv2.getTrackbarPos("L-v","track")
        uh=cv2.getTrackbarPos("h-H","track")
        us=cv2.getTrackbarPos("h-s","track")
        uv=cv2.getTrackbarPos("h-v","track")


        lb=np.array([lh,ls,lv])
        ub=np.array([uh,us,uv])

        

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(roi, -1, disc, roi)
        # min_YCrCb = np.array([0,133,77],np.uint8)
        # max_YCrCb = np.array([235,173,127],np.uint8)
        # imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        # skinRegionYCrCb = cv2.inRange(imageYCrCb,lb,ub)
        # skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)
        # mask=cv2.inRange(hsv,lb,ub)
        mask=cv2.GaussianBlur(roi,(3,3),0)
        kernel=np.ones((5,5),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel,iterations=1)
        
        cv2.imshow('mask1',mask)
        return mask
while True :
    
        ret,frame=video.read()
        frame=cv2.flip(frame,1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        getFaces(gray,frame)
        s=frame[20:360,20:360]
        hsv=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)
        
        cv2.rectangle(frame,(20,20),(360,300),(0,255,0),3)
        #obj=cv2.cvtColor(obj,cv2.COLOR_BGR2HSV)
        #objHist=cv2.calcHist([obj],[0,1],None,[180,256],[0,180,0,256])
        roi=cv2.calcBackProject([hsv],[0,1],object_hist,[0,180,0,256],1)
        
        _, roi = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY)

        

        mask=filterImg(roi)

        con,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        
        try :
            con=max(con,key=lambda x : cv2.contourArea(x))
        #hull=[cv2.convexHull(i,returnPoints=False) for i in con ]
            #hull=cv2.convexHull(con)
            if cv2.contourArea(con)>11000:
                #cv2.drawContours(s,[hull],-1,(200,0,0),5)
                hull = cv2.convexHull(con,returnPoints = False)
                defects = cv2.convexityDefects(con,hull)
                cd=0
                M = cv2.moments(con)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if not isUP :
                    initialPOS=[cX,cY]
                    isUP=True
                cv2.circle(s,(cX,cY),5,[0,0,255],-1)
                for i in range(defects.shape[0]):
                    S,e,f,d = defects[i,0]
                    start = tuple(con[S][0])
                    end = tuple(con[e][0])
                    far = tuple(con[f][0])

                    a= math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
                    
                    if angle <= 90:
                        cd += 1

                    cv2.line(s,start,end,[0,255,0],2)
                
                if cd== 0:
                    cv2.putText(frame,"F/R", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2) 
                    if cX-initialPOS[0]>=50 :
                        pyautogui.hotkey('ctrl','right')
                        
                    elif initialPOS[0]-cX>=50 :
                        pyautogui.hotkey('ctrl','left')
                        
                    
                elif cd == 1:      
                    cv2.putText(frame,"Volume", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    mute=False
                    cv2.circle(s,end,5,[0,0,255],-1)
                    if initialPOS[1]-cY>=30 :
                        pyautogui.hotkey('ctrl','up')
                        
                    elif cY-initialPOS[1]>=40 :
                        pyautogui.hotkey('ctrl','down')
                        
                    
                elif cd == 2:
                    cv2.putText(frame,"Mute", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    if mute==False :
                        pyautogui.press('m')
                        mute=True
                elif cd == 3:
                    cv2.putText(frame,"PLAY", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    if pause==True :
                        pyautogui.press('space')
                        pause=False

                elif cd == 4:
                    cv2.putText(frame,"Pause", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    #os.system("vlc-ctrl volume -10%")
                    if pause==False :
                        pyautogui.press('space')
                        pause=True
                else:
                    pass
            else :
                isUP=False;
        except:
            pass

            
        
        cv2.imshow('hsv',hsv)
        cv2.imshow('img',frame)
        
    
        #k=cv2.waitKey(1)
        if cv2.waitKey(1)==ord('q') :
            break

video.release()
cv2.destroyAllWindows()