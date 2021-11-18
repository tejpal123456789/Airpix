import google
from Google import Create_Service
#CLIENT_SECRET_FILE='client_secrets.json'
#API_NAME='drive'
#API_VERSION='v3'
#SCOPE=['https://www.googleapis.com/auth/drive']
#service=Create_Service(CLIENT_SECRET_FILE,API_NAME,API_VERSION,SCOPE)
#print(dir(service))

#fields=['Data Science','Machine Learning','Deep Learning']

#for field in fields:
   # file_metadata={'name':field,
         #           'mimeType':'application/vnd.google-apps.folder'}

  #  service.files().create(body=file_metadata).execute()

import os
folder= '1WXPKKkrtqFIP5bNPlLqWBjdnrUoB5ut4'
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
gauth=GoogleAuth()
drive=GoogleDrive(gauth)

#file1=drive.CreateFile({'parents':[{'id':folder}],'title':'hello.txt'})
#file1.SetContentString('hello world !!!!')
#file1.Upload()

directory='C:/Users/TEJPAL KUMAWAT/OneDrive/Desktop/Airpix'
for f in os.listdir(directory):
    filepath=os.path.join(directory,f)
    #print(filepath)
    gfile=drive.CreateFile({'parents':[{'id':folder}],'title':f})
    gfile.SetContentFile(filepath)
    gfile.Upload()
    


