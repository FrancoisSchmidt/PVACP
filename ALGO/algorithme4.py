from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def tr(c, o, coeff):
    return(int((c-o)*coeff)+o)

def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[0],mouth[6])
	B = dist.euclidean(mouth[3],mouth[-1])
	mar = B/A
	return mar


EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 7
MOUTH_AR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 7

EYE_CLOSE_FRAMES = 10
OPEN_COUNTER_right = 0
OPEN_COUNTER_left = 0
COUNTER_right = 0
COUNTER_left = 0 
OEIL=False

cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print((lStart, lEnd),(rStart, rEnd))


while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	tickmark = cv2.getTickCount()
	faces=detector(gray, 0)
	if faces is not None:
		i=np.zeros(shape=(frame.shape), dtype=np.uint8)
	for face in faces:
		x1=face.left()
		y1=face.top()
		x2=face.right()
		y2=face.bottom()

		landmarks=predictor(gray, face)

		d_eyes=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(45).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(45).y, 2))
		d1=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(30).y, 2))
		d2=math.sqrt(math.pow(landmarks.part(45).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(45).y-landmarks.part(30).y, 2))
		coeff=d1+d2

		a1=int(250*(landmarks.part(36).y-landmarks.part(45).y)/coeff)
		a2=int(250*(d1-d2)/coeff)
		cosb=min((math.pow(d2, 2)-math.pow(d1, 2)+math.pow(d_eyes, 2))/(2*d2*d_eyes), 1)
		a3=int(250*(d2*math.sin(math.acos(cosb))-coeff/4)/coeff)

		for n in range(0, 68):
			x=landmarks.part(n).x
			y=landmarks.part(n).y
			cv2.circle(frame, (x,y), 3, (255,0,0), -1)
			if n==30 or n==36 or n==45:
				cv2.circle(i, (x, y), 3, (255, 255, 0), -1)
			else:
				cv2.circle(i, (x, y), 3, (255, 0, 0), -1)
	
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)
		mouth = shape[48:58]

		cv2.circle(frame ,(mouth[0][0],mouth[0][1]),3, (0, 0, 255), -1)
		cv2.circle(frame ,(mouth[-1][0],mouth[-1][1]),3, (0, 0, 255), -1)
		cv2.circle(frame ,(mouth[3][0],mouth[3][1]),3, (0, 0, 255), -1)
		cv2.circle(frame ,(mouth[6][0],mouth[6][1]),3, (0, 0, 255), -1)
		mar = mouth_aspect_ratio(mouth)
		print(mar)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)



		if leftEAR < EYE_AR_THRESH:
			COUNTER_left += 1
			OPEN_COUNTER_left = 0
			cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
			if COUNTER_left >= EYE_AR_CONSEC_FRAMES:
				OEIL=True	

		else:
			OPEN_COUNTER_left += 1
			if OEIL == True and OPEN_COUNTER_left < EYE_CLOSE_FRAMES:
				cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				OEIL=True

			else:
				cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (10, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				COUNTER_left = 0
				OEIL=False



		if rightEAR < EYE_AR_THRESH:
			COUNTER_right += 1
			OPEN_COUNTER_right = 0
			cv2.putText(frame, "EAR: {:.2f}".format(rightEAR), (300, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
			if COUNTER_right >= EYE_AR_CONSEC_FRAMES:
				OEIL=True
				
		else:
			OPEN_COUNTER_right += 1
			if OEIL == True and OPEN_COUNTER_right < EYE_CLOSE_FRAMES:
				cv2.putText(frame, "EAR: {:.2f}".format(rightEAR), (300, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				OEIL=True
			else:
				cv2.putText(frame, "EAR: {:.2f}".format(rightEAR), (300, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				COUNTER_right = 0
				OEIL=False

		
	
		flag=1
		txt="Franz ne fait rien "
		if a1<-40 and OEIL==True:
			txt="Franz veut tourner a gauche "
			print("Gauche")
		if a1>40 and OEIL==True:
			txt="Franz veut tourner a droite "
			print("Droite")


		cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
	cv2.imshow("Frame", frame)
	key=cv2.waitKey(1)&0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
