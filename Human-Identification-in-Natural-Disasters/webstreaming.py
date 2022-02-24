import cv2
from imutils.object_detection import non_max_suppression
import csv
import imutils
from flask import Flask, render_template, Response
import requests
import numpy as np
import random
from geopy.geocoders import Nominatim
ls=[]
loc=[]
count=[]
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
geolocator = Nominatim(user_agent="People count")
app = Flask(__name__)
print("[INFO] accessing video stream...")
video_capture = cv2.VideoCapture('video.mp4')

#preprocessing the frame captured from camera
def detect(image):
    try:
        
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
    	# detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
    		padding=(8, 8), scale=1.05)
    	# draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    	# apply non-maxima suppression to the bounding boxes using a
    	# fairly large overlap threshold to try to maintain overlapping
    	# boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        #print(len(pick))
        ls.append(len(pick))
        #print(ls)

        
    	# draw the final bounding boxes
        
        for (xA, yA, xB, yB) in pick:
            da=cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            location = geolocator.geocode("Hyderabad")
            print(location.latitude, location.longitude)
            loc.append(location.latitude)
            loc.append(location.longitude)
            cv2.imwrite('Output/image'+str(random.randint(1, 1000))+'.jpg',da)
            with open('Output/location.csv', mode='a',newline="") as csv_file:
                writer = csv.writer(csv_file,delimiter=',')
                writer.writerow(loc)
                loc.clear()
        return image
    except Exception as e:
        one=int((ls.count(1))/6)
        two= int((ls.count(2))/6)
        three= int((ls.count(3))/6)
        four= int((ls.count(4))/6)
        count.extend([one,two,three,four])
        people=int(sum(count)/len(count))
        print(people)
        url="https://www.fast2sms.com/dev/bulk?authorization=enter your api key&sender_id=FSTSMS&message="+str(people)+" people are in danger.Need to rescue them&language=english&route=p&numbers="+str(your number)
        result=requests.request("GET",url)
        print(e)
    


@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        _,frame = video_capture.read()
        image= detect(frame)
        '''if not(np.shape(frame) == ()):
            cv2.imshow("Human Identifier", image)'''
       # if cv2.waitKey(1) & 0xFF == ord('q') or np.shape(frame) == ():
        #    break
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
        bytearray(encodedImage) + b'\r\n')
       


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
