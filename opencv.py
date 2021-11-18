import cv2
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output1.avi',fourcc,20.0,(640,480))

cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    cv2.imshow('Video',frame)
    out.write(frame)


    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



