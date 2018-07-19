import numpy as np
import cv2

a = np.zeros((256,256,3))

for i in range(0,256):
    for j in range(0,256):
        a[i][j][0]=1
        a[i][j][1]=i/255
        a[i][j][2]=1-i/255
#print(a)
cv2.imshow('img', a)
