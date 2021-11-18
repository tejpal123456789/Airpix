


import cv2
fourcc=cv2.VideoWriter_fourcc(*'XVID')
save_path='output1.avi'
out=cv2.VideoWriter(save_path,fourcc,20.0,(640,480))

cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    cv2.imshow('Video',frame)
    out.write(frame)


    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


import os
folder= '1WXPKKkrtqFIP5bNPlLqWBjdnrUoB5ut4'
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
gauth=GoogleAuth()
drive=GoogleDrive(gauth)
#file1=drive.CreateFile({'parents':[{'id':folder}],'title':'CV DATA'})
#file1.SetContentFile(save_path)
#file1.Upload()
directory='C:/Users/TEJPAL KUMAWAT/google drive api'
for f in os.listdir(directory):
    print(f)
    filepath=os.path.join(directory,f)
    #print(filepath)
   # gfile=drive.CreateFile({'parents':[{'id':folder}],'title':f})
   # gfile.SetContentFile(filepath)
   # gfile.Upload()

