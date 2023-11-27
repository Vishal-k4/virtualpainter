import cv2
import time
import mediapipe as mp

class HandDetection():
    def __init__(self,mode=False,maxhands=2,model_complexity=1,detectioncon=0.5,trackingcon=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.model_complexity=model_complexity
        self.detectioncon=detectioncon
        self.trackingcon=trackingcon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxhands, self.model_complexity,self.detectioncon,self.trackingcon)
        self.mpDraw=mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]

    def findhands(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgrgb)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,hand,self.mpHands.HAND_CONNECTIONS)
        return img

    def getpos(self,img,handnumber=0,draw=True):
        self.lmlst=[]

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handnumber]
            for id,lm in enumerate(myhand.landmark):
                 h,w,c=img.shape
                 cx,cy=int(lm.x*w),int(lm.y*h)
                 self.lmlst.append([id,cx,cy])
                 if draw:
                     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return self.lmlst

    def fingersup(self):
        fingers=[]
        #Thumb
        if self.lmlst[self.tipIds[0]][1]<self.lmlst[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #4 fingers
        for id in range(1,5):
            if self.lmlst[self.tipIds[id]][2]<self.lmlst[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    Ptime=0
    Ctime=0
    cap=cv2.VideoCapture(0)
    detector=HandDetection()


    while True:
        sucess,img=cap.read()
        img=detector.findhands(img)
        lmlst = detector.getpos(img)
        if len(lmlst) != 0:
            print(lmlst[0])
        Ctime=time.time()
        fps=1/(Ctime-Ptime)
        Ptime=Ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('Image',img)
        cv2.waitKey(1)



if __name__=="__main__":
    main()
