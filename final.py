import cv2,sys,numpy,random,os,math,select,time


fn_dir='faces'

images = []
lables = []


names = {}
colours={}
id=0
for subdir in os.listdir(fn_dir):
 if subdir[0]=='.':
  continue
 names[id]=subdir
 colours[id]=(random.randrange(256),random.randrange(256),random.randrange(256))
 subjectpath=os.path.join(fn_dir, subdir)
 for filename in os.listdir(subjectpath):
  if filename[0]=='.':
   continue
  path=os.path.join(subjectpath, filename)
  lable=id
  images.append(cv2.imread(path,0))
  lables.append(int(lable))
 id+=1

im_width=images[0].shape[0]
im_height=images[0].shape[1]

model = cv2.createFisherFaceRecognizer()
print "Loading Trained data.."
model.load("training_data.xml")
print "Data loaded"

def save_person(name):
	with open('people.txt', 'a+') as f:
		f.write(time.strftime('%X') + ' ' + name  + '\n')






faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame=cv2.flip(frame,1,0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize=cv2.resize(face,(im_width, im_height))
        prediction=model.predict(face_resize)
        print prediction
        print names[prediction[0]]
        cv2.rectangle(frame, (x,y), (x+w, y+h), colours[prediction[0]], 3)
        if(prediction[1]>299):
        	cv2.putText(frame, '%s %.0f'%(names[prediction[0]], prediction[1]) + '%', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, colours[prediction[0]])
        	save_person(names[prediction[0]])
  
        else:
        	cv2.putText(frame, '%s'%("Unknown"), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, colours[prediction[0]])


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()