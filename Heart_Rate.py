import cv2
import time
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Change these variables based on the location of your cascade classifier
PATH_TO_HAAR_CASCADES = "image_processing/" 
face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used
firstFrame = None
time = []
R = []
G = []
B = []
pca = FastICA(n_components=3)
cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0
plt.ion()
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame_num += 1
        if firstFrame is None:
            start = datetime.datetime.now()
            time.append(0)
            # Take first frame and find face in it
            firstFrame = frame
            cv2.imshow("frame",firstFrame)
            old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) 
            if faces == ():
                firstFrame = None
            else:
                for (x,y,w,h) in faces: 
                    x2 = x+w
                    y2 = y+h
                    cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.imshow("frame",firstFrame)
                    VJ_mask = np.zeros_like(firstFrame)
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                ROI = VJ_mask
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask)
                cv2.imshow('ROI',ROI_color)
                R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI)
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        else:
            current = datetime.datetime.now()-start
            current = current.total_seconds()
            time.append(current)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ROI_color = cv2.bitwise_and(frame,frame)
            cv2.imshow('ROI',ROI_color)
            R_new,G_new,B_new,_ = cv2.mean(ROI_color)
            R.append(R_new)
            G.append(G_new)
            B.append(B_new)
            if frame_num >= 900:
                N = 900
                G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1))
                G_std = G_std.reshape(1, -1)[0]
                R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1))
                R_std = R_std.reshape(1, -1)[0]
                B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                B_std = B_std.reshape(1, -1)[0]
                T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)]))
                X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose()
                N = len(X_f[0])
                yf = fft(X_f[1])
                yf = yf/np.sqrt(N)
                xf = fftfreq(N, T)
                xf = fftshift(xf)
                yplot = fftshift(abs(yf))
                plt.figure(1)
                plt.gcf().clear()
                fft_plot = yplot
                fft_plot[xf<=0.75] = 0
                
                #print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm')
                plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)])
                plt.pause(0.0001)
