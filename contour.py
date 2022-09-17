   import cv2
import numpy as np
import math
import os
import subprocess
import pyautogui
#pyautogui.PAUSE = 0.0
video=cv2.VideoCapture(0)
cv2.namedWindow('track')



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
while True :
    
        ret,frame=video.read()
        frame=cv2.flip(frame,1)
        s=frame[100:450,100:450]
        hsv=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame,(100,100),(450,450),(0,255,0),3)
        faces=face.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=7)
        for x,y,w,h in faces :
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            frame[y:y+h,x:x+w]=0
        lh=cv2.getTrackbarPos("L-H","track")
        ls=cv2.getTrackbarPos("L-s","track")
        lv=cv2.getTrackbarPos("L-v","track")
        uh=cv2.getTrackbarPos("h-H","track")
        us=cv2.getTrackbarPos("h-s","track")
        uv=cv2.getTrackbarPos("h-v","track")


        lb=np.array([lh,ls,lv])
        ub=np.array([uh,us,uv])

        

        
        # min_YCrCb = np.array([0,133,77],np.uint8)
        # max_YCrCb = np.array([235,173,127],np.uint8)
        # imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        # skinRegionYCrCb = cv2.inRange(imageYCrCb,lb,ub)
        # skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)
        mask=cv2.inRange(hsv,lb,ub)
        mask=cv2.GaussianBlur(mask,(3,3),0)
        kernel=np.ones((3,3),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel,iterations=1)
        
        res=cv2.bitwise_and(s,s,mask=mask)
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
                cv2.circle(s,(cX,cY),5,[0,0,255],-1)
                for i in range(defects.shape[0]):
                    S,e,f,d = defects[i,0]
                    start = tuple(con[S][0])
                    end = tuple(con[e][0])
                    far = tuple(con[f][0])

                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
                    
                    if angle <= 90:
                        cd += 1
                        #cv2.circle(s,end,5,[0,0,255],-1)

                    cv2.line(s,start,end,[0,255,0],2)
                
                if cd== 0:
                    cv2.putText(frame,"F/R", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2) 
                    #cv2.circle(s,(cX,cY),5,[0,0,255],-1)
                    if cX>=250 :
                        pyautogui.hotkey('ctrl','right')
                        
                    elif cX<=100 :
                        pyautogui.hotkey('ctrl','left')
                        
                    
                elif cd == 1:      
                    cv2.putText(frame,"Volume", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    #os.system("vlc-ctrl pause")

                    cv2.circle(s,end,5,[0,0,255],-1)
                    if cY<=130 :
                        pyautogui.hotkey('ctrl','up')
                        mute=False
                    elif cY>=240 :
                        pyautogui.hotkey('ctrl','down')
                        mute=False
                    
                elif cd == 2:
                    cv2.putText(frame,"Mute", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    if mute==False :
                        pyautogui.press('m')
                        mute=True
                    #os.system("vlc-ctrl volume 0")
                elif cd == 3:
                    cv2.putText(frame,"PLAY", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
                    ##os.system("vlc-ctrl volume +10%")
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
        except:
            pass

            
        
        cv2.imshow('hsv',mask)
        cv2.imshow('img',frame)
        cv2.imshow('th',hsv)
    
        #k=cv2.waitKey(1)
        if cv2.waitKey(1)==ord('q') :
            break

video.release()
cv2.destroyAllWindows()