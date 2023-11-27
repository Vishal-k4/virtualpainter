import cv2
import numpy as np
import time
#import os
import handtrackingmodule as htm
# folderpath='Header'
# myList=os.listdir(folderpath)
# overlst=[]
# for impath in myList:
#     image=cv2.imread(f'{folderpath}/{impath}')
#     overlst.append(image)
# header=overlst[0]

brush=15
eraser=50

drawcolor=(255,0,255)
cap=cv2.VideoCapture(0)
cap.set(3,1288)
cap.set(4,720)

detector=htm.HandDetection(detectioncon=0.85)
xp,yp=0,0
imgcanvas=np.zeros((720,1280,3),np.uint8)
while True:
    sucess,img=cap.read()
    img=cv2.flip(img,1)

    cv2.circle(img,(200,30),15,(255,0,255))
    cv2.circle(img,(500,30),15,(255,255,255))
    cv2.circle(img,(700,30),15,(0,0,255))
    cv2.circle(img,(1000,30),20,(0,0,0),3)

    img=detector.findhands(img)
    lmlst=detector.getpos(img,draw=False)
    if len(lmlst) != 0:
        #print(lmlst)
        x1,y1=lmlst[8][1:]
        x2,y2=lmlst[12][1:]
        fingers=detector.fingersup()
        if fingers[1] and fingers[2] :
            xp,yp=0,0
            print("selection mode")
            if y1<125:
                if x1<300 :
                    cv2.circle(img,(200,30),15,(255,0,255),cv2.FILLED)
                    drawcolor=(255,0,255)
                elif 400<x1<600 :
                    cv2.circle(img,(500,30),15,(255,255,255),cv2.FILLED)
                    drawcolor=(255,255,255)
                elif 650<x1<800:
                    cv2.circle(img,(700,30),15,(0,0,255),cv2.FILLED)
                    drawcolor=(0,0,255)
                elif 850<x1<1100:
                    cv2.circle(img,(1000,30),20,(0,0,0),cv2.FILLED)
                    drawcolor=(0,0,0)
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drawcolor,cv2.FILLED)
        if fingers[1] and fingers[2]==False :
            cv2.circle(img,(x1,y1),15,drawcolor,cv2.FILLED)
            print("drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawcolor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawcolor,eraser)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,eraser)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawcolor,brush)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,brush)
            xp,yp=x1,y1
    imggray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    _,imginv=cv2.threshold(imggray,50,255,cv2.THRESH_BINARY_INV)
    imginv=cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imginv)
    img=cv2.bitwise_or(img,imgcanvas)

    # img=cv2.addWeighted(img,0.5,imgcanvas,0.5,0)
    cv2.imshow('Image',img)
    #cv2.imshow('canvas',imgcanvas)
    cv2.waitKey(1)
