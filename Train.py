import cv2,sys,numpy,random,os,math,select,time
def train():
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


 images = numpy.array(images)
 lables = numpy.array(lables)

 print "Training.."


 model = cv2.createLBPHFaceRecognizer()
 model.train(images, lables)
 model.save("training_data.xml")


train()
