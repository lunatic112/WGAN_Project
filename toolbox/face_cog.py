import cv2
import sys
import os.path
import tqdm

counter=0

def detect(filename, cascade_file = "face_cog.xml", counter=0):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.imwrite(f".\\out\\{counter}.png", image[y:y+h,x:x+w])
        counter+=1
    
    return counter

#reading the file list
with open(sys.argv[1], 'r') as f:
    pic_list=f.readlines()

counter=0
for i in tqdm(pic_list):
    if i[-1]=='\n':
        i = i[:-1]
    counter=detect(i,counter=counter)