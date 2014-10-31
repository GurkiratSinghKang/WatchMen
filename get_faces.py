import cv2,sys,numpy,csv
import os

def main():
 fn_haar='haarcascade_frontalface_alt.xml'
 fn_dir='faces'
 fn_name=sys.argv[1]
 path = os.path.join(fn_dir, fn_name)

 if os.path.isdir(path):
  print('Name already exists.')
  cont=raw_input('Continue? y/N:').lower()
  if cont!='y':
   return 0
  id=len([name for name in os.listdir(path) if os.path.isfile(name)])
 else:
  os.mkdir(path)
  id=1


 
 downsize = 4

 im_width, im_height = 92, 112
 #im_width, im_height = 100,100

 haar_cascade = cv2.CascadeClassifier(fn_haar)
 webcam = cv2.VideoCapture(0)

 while True:
  if id>250:
   break
  rval, frame = webcam.read()
  frame=cv2.flip(frame,1,0)
  gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  minigray = cv2.resize(gray, (gray.shape[1]/downsize,gray.shape[0]/downsize))
  faces = haar_cascade.detectMultiScale(minigray)
  faces = sorted(faces,key=lambda x: x[3])
  if faces:
   face_i=faces[0]
   x,y,w,h=[v*downsize for v in face_i]
   face=gray[y:y+h, x:x+w]
   face_resize=cv2.resize(face,(im_width, im_height)) 
   cv2.imwrite('%s/%s.png'%(path, id), face_resize)
   cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 3)
   pos_x=x-10
   pos_y=y-10
   cv2.putText(frame, '%s'%(fn_name), (pos_x, pos_y), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
   id +=1
  cv2.imshow("faces", frame)
  key=cv2.waitKey(10)
  if key==27:
   break


main()

