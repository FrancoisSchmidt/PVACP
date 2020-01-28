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

#Eye Aspect Ratio pour savoir quand on cligne des yeux
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

#Mouth aspect Ratio pour savoir quand on ouvre la bouche
def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[0], mouth[6])
	B = dist.euclidean(mouth[3], mouth[-1])
	mar = B / A
	return mar

def dec2bin(d,nb=8):
    """Représentation d'un nombre entier en binaire"""
    if d == 0:
        return "0".zfill(nb)
    if d<0:
        d += 1<<nb
    b=""
    while d != 0:
        d, r = divmod(d, 2)
        b = "01"[r] + b
    return b.zfill(nb)


EYE_AR_THRESH = 0.26			#Constante en-deça de laquelle l'oeil est considéré comme fermé
EYE_AR_CONSEC_FRAMES = 5		#Nombre de frames minimum pour considérer l'oeil comme fermé
MOUTH_AR_THRESH = 0.5			#Constante au dela de laquelle la bouche est considérée comme ouverte
MOUTH_AR_CONSEC_FRAMES = 5		#Nombre de frames minimum pour considérer la bouche comme ouverte

EYE_CLOSE_FRAMES = 5			#Nombre de frames pendant lesquelles l'oeil est encore considéré comme fermé
OPEN_COUNTER_right = 0			#Compteur oeil droit
OPEN_COUNTER_left = 0			#Compteur oeil gauche
COUNTER_right = 0
COUNTER_left = 0 
OEIL=False						#Oeil=False s'il est ouvert, et True s'il est fermé
OEIL_gauche = False
OEIL_droit = False
Commande = 0

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()									#Importation du face_detector de dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")		#Importation de la base de données de visages

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]		#Coordonnées de l'oeil gauche
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]		#Coordonnées de l'oeil droit
compteur_frame = 0

while True:
	Commande = None
	compteur_frame += 1
	ret, frame = cap.read()						#Leture de frame
	frame = imutils.resize(frame, width=450)	#Resize
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	tickmark = cv2.getTickCount()				#Nombre de ticks
	faces=detector(gray, 0)
	if faces is not None:
		i=np.zeros(shape=(frame.shape), dtype=np.uint8)	#Si pas de visage détécté, Matrice de zéros
		#txt = str(compteur_frame)+"  No face found"
	for face in faces:
		landmarks=predictor(gray, face)

		#Calcul de coefficients pour l'inclinaison de la tête.
		d_eyes=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(45).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(45).y, 2))
		d1=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(30).y, 2))
		d2=math.sqrt(math.pow(landmarks.part(45).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(45).y-landmarks.part(30).y, 2))
		coeff=d1+d2
		#Calcul de coefficients pour l'inclinaison de la tête.
		a1=int(250*(landmarks.part(36).y-landmarks.part(45).y)/coeff)
		a2=int(250*(d1-d2)/coeff)
		cosb=min((math.pow(d2, 2)-math.pow(d1, 2)+math.pow(d_eyes, 2))/(2*d2*d_eyes), 1)
		a3=int(250*(d2*math.sin(math.acos(cosb))-coeff/4)/coeff)

		for n in range(0, 68):
			x=landmarks.part(n).x
			y=landmarks.part(n).y
			cv2.circle(frame, (x,y), 3, (255,0,0), -1)
			if n==30 or n==36 or n==45:
				cv2.circle(i, (x, y), 3, (255, 255, 0), -1)	#Dessin du contour du visage
			else:
				cv2.circle(i, (x, y), 3, (255, 0, 0), -1)
	
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)
		mouth = shape[48:58]		#Coordonées de la bouche
		cv2.circle(frame ,(mouth[0][0],mouth[0][1]),3, (0, 0, 255), -1)		#Marque les contours de la bouche
		cv2.circle(frame ,(mouth[-1][0],mouth[-1][1]),3, (0, 0, 255), -1)
		cv2.circle(frame ,(mouth[3][0],mouth[3][1]),3, (0, 0, 255), -1)
		cv2.circle(frame ,(mouth[6][0],mouth[6][1]),3, (0, 0, 255), -1)
		mar = mouth_aspect_ratio(mouth)				#Calcule le mouth aspect ratio

		leftEye = shape[lStart:lEnd]				#Récupère les coordonnées de l'oeil gauche
		rightEye = shape[rStart:rEnd]				#Récupère les coordonnées de l'oeil droite
		leftEAR = eye_aspect_ratio(leftEye)			#Calcule le eye aspect ratio oeil gauche
		rightEAR = eye_aspect_ratio(rightEye)		#De même oeil droit
		ear = (leftEAR + rightEAR) / 2.0			#Moyenne des deux yeux
		leftEyeHull = cv2.convexHull(leftEye)		
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)	#Dessin du contour des yeux
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


		if leftEAR < EYE_AR_THRESH:		#Si l'oeil gauche est considéré comme fermé
			COUNTER_left += 1			#Incrémentation du compteur de frame
			OPEN_COUNTER_left = 0		
			if COUNTER_left >= EYE_AR_CONSEC_FRAMES:	#Si l'oei est fermé depuis "EYE_AR_CONSEC_FRAMES", alors affiché qu'il est fermé
				OEIL=True	#Oeil fermé
				OEIL_gauche=True
		else:
			OPEN_COUNTER_left += 1		
			if OEIL == True and OPEN_COUNTER_left < EYE_CLOSE_FRAMES:	#Si l'oeil était fermé et qu'il n'est ouvert que depuis EYE_CLOSE_FRAMES frames, le considérer encore comme fermé
				OEIL=True
				OEIL_gauche=True

			else:			#Sinon, le considérer comme ouvert
				COUNTER_left = 0
				OEIL=False
				OEIL_gauche=False


		if rightEAR < EYE_AR_THRESH:	#Même chose avec l'oeil droit
			COUNTER_right += 1
			OPEN_COUNTER_right = 0
			if COUNTER_right >= EYE_AR_CONSEC_FRAMES:
				OEIL=True
				OEIL_droit=True
		else:
			OPEN_COUNTER_right += 1
			if OEIL == True and OPEN_COUNTER_right < EYE_CLOSE_FRAMES:
				OEIL=True
				OEIL_droit=True
			else:
				COUNTER_right = 0
				OEIL=False
				OEIL_droit=False

		txt = str(compteur_frame)+"  "
			
			
		if OEIL==True and mar > MOUTH_AR_THRESH and a1<40 and a1>-40:	#Si l'oei est fermé et la bouche ouverte
			Commande = 55
		else:
			flag=1
			if a1<-40 and OEIL==True:		#Si la tête est inclinée vers la gauche et au moins un oeil est fermé
				Commande = -127

			if a1>40 and OEIL==True:		#Si la tête est inclinée vers la droite et au moins un oeil est fermé
				Commande = 127


	print(txt, Commande)
	key=cv2.waitKey(1)&0xFF
	if key==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
