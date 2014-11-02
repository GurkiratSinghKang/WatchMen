import cv2,sys,numpy,random,os,math,select,time
from threading import Thread
import sqlite3,datetime,sys


def prompt() :
 sys.stdout.write('\rWatchMen>>  ')
 sys.stdout.flush()



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

conn = sqlite3.connect('Watchmen.db')
conn.execute("""CREATE TABLE IF NOT EXISTS People
       (ID INTEGER  PRIMARY KEY    NOT NULL,
       NAME           TEXT    NOT NULL,
       CAMERA_NO            TEXT     NOT NULL,
       LAST_SEEN_TIME          TEXT    NULL );""")
print "Table created successfully";

conn.close()



def save_person(person_id,person_name,camera):
    if(str(camera) == '0'):
      camera_no = 0
    pos = str(datetime.datetime.now()).find('.')
    time_now=datetime.datetime.strptime(str(datetime.datetime.now())[:pos] , '%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect('Watchmen.db')
    #cur2 = conn.execute("SELECT * from PEOPLE WHERE NAME=:name",{"name":str(person_name)})
    cur2 = conn.execute("SELECT Max(ID) FROM PEOPLE WHERE NAME='%s';"%(str(person_name)));
    max_id = cur2.fetchone()[0]
    if(max_id != None):
      cur2 = conn.execute("SELECT LAST_SEEN_TIME, CAMERA_NO from PEOPLE WHERE NAME=:name and ID=:id",{"name":str(person_name),"id":int(max_id)})
      need = cur2.fetchone()
      last_known_time = need[0]

      last_known = datetime.datetime.strptime(last_known_time , '%Y-%m-%d %H:%M:%S')
      threshold = datetime.datetime.strptime("0:05:00" , '%H:%M:%S').time()
      
      time_diff = datetime.datetime.strptime(str(time_now-last_known) , '%H:%M:%S').time()
      if ((time_diff)>threshold):
        conn.execute("INSERT INTO PEOPLE (NAME,CAMERA_NO,LAST_SEEN_TIME) VALUES (?,?,?)",(str(person_name),str(camera),str(time_now)));
        conn.commit()
      else:
        if(int(need[1])==camera_no):
          cursor = conn.execute("SELECT Max(ID) FROM PEOPLE WHERE NAME='%s';"%(str(person_name)));
          max_id = cursor.fetchone()[0]
          conn.execute("UPDATE PEOPLE SET LAST_SEEN_TIME=? WHERE id=?", (time_now, int(max_id)))
          conn.commit()
        else:
          conn.execute("INSERT INTO PEOPLE (NAME,CAMERA_NO,LAST_SEEN_TIME) VALUES (?,?,?)",(str(person_name),str(camera),str(time_now)));
          conn.commit()
    else:
      conn.execute("INSERT INTO PEOPLE (NAME,CAMERA_NO,LAST_SEEN_TIME) VALUES (?,?,?)",(str(person_name),str(camera),str(time_now)));
      conn.commit()

    conn.close()






faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")



def main(cam):
    video_capture = cv2.VideoCapture(cam)

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
            cv2.rectangle(frame, (x,y), (x+w, y+h), colours[prediction[0]], 3)
            if(prediction[1]>299):
                cv2.putText(frame, '%s %.0f'%(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, colours[prediction[0]])
                save_person(prediction[0],names[prediction[0]],cam)
      
            else:
                cv2.putText(frame, '%s'%("Unknown"), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, colours[prediction[0]])


        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()




class StartCam(Thread):
    def __init__(self, cam):
        ''' Constructor. '''
 
        Thread.__init__(self)
        self.cam = cam
 
 
    def run(self):
        main(self.cam)


cam1 = StartCam(0)
cam1.start()


choice = 0
while(int(choice)<4):
  prompt()
  print "Press any key  "
  raw_input()
  prompt()
  print "What do you wanna do?"
  choice = raw_input("\t******MENU******\n1. Last Known Location \n2. Track \n3. Notify when Next seen\n4. Exit\n\t=>  ")
  if (int(choice)==1):
    prompt()
    person = raw_input("Enter Name:  ")
    conn = sqlite3.connect('Watchmen.db')
    cur2 = conn.execute("SELECT Max(ID) FROM PEOPLE WHERE NAME='%s';"%(str(person)));
    max_id = cur2.fetchone()[0]
    if(max_id != None):
      cur2 = conn.execute("SELECT LAST_SEEN_TIME, CAMERA_NO from PEOPLE WHERE NAME=:name and ID=:id",{"name":str(person),"id":int(max_id)})
      need = cur2.fetchone()
      last_known_time = need[0]
      prompt()
      print need[0]
    else:
      prompt()
      print "Sorry! Person not in database."

  if (int(choice)==2):
    prompt()
    person = raw_input("Enter Name:  ")
    conn2 = sqlite3.connect('Watchmen.db')
    cur2 = conn2.execute("SELECT * from PEOPLE WHERE NAME=:name",{"name":str(person)})
    rows = cur2.fetchall()
    if(rows==None):
      prompt()
      print "Sorry! Person not in database."
    else:
      prompt()
      for row in rows:
        print row[3]
  

  if (int(choice)==4):
       sys.exit(0)





cam1.join()