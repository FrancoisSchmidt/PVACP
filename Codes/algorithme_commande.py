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
OEIL=False						#Oeil=True si au moins un des deux yeux est fermé 
OEIL_gauche = False				#Même chose que OEIL, mais seulement pour le gauche
OEIL_droit = False				#Même chose mais pour le droit
Commande = 0

cap = cv2.VideoCapture(0)		#Acquisition du flux vidéo provenant de la webcam

detector = dlib.get_frontal_face_detector()										#Importation du face_detector de dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")		#Importation de la base de données de visages

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]		#Coordonnées de l'oeil gauche
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]		#Coordonnées de l'oeil droit
compteur_frame = 0													#Compteur de frames écoulées depuis le début de l'algorithme

while True:
	Commande = None									#Pas de commande
	compteur_frame += 1								#On incrémente le numéro de frame
	ret, frame = cap.read()							#Leture de frame
	frame = imutils.resize(frame, width=450)		#Resize (utile seulement pour l'affichage avec cv2, à supprimer à la fin)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	#Conversion de l'image en nuances de noir et blanc
	#tickmark = cv2.getTickCount()					#Nombre de ticks
	faces=detector(gray, 0)							#Recherche de visages
	if faces is not None:
		i=np.zeros(shape=(frame.shape), dtype=np.uint8)	#Si pas de visage détécté, Matrice de zéros
		txt = str(compteur_frame)+"  No face found"
	for face in faces:								#Pour tous les visages trouvés (en l'occurence un seul, celui du pilote, donc boucle for itérée une seule fois)
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


		shape = predictor(gray, face)				#Predictor appliqué sur le visage
		shape = face_utils.shape_to_np(shape)		#Conversion en matrice en fonction des points prédéfinis (voir doc dlib)
		mouth = shape[48:58]						#Coordonées de la bouche
		mar = mouth_aspect_ratio(mouth)				#Calcule le mouth aspect ratio

		leftEye = shape[lStart:lEnd]				#Récupère les coordonnées de l'oeil gauche
		rightEye = shape[rStart:rEnd]				#Récupère les coordonnées de l'oeil droite
		leftEAR = eye_aspect_ratio(leftEye)			#Calcule le eye aspect ratio oeil gauche
		rightEAR = eye_aspect_ratio(rightEye)		#Calcule le eye aspect ratio oeil droit
		ear = (leftEAR + rightEAR) / 2.0			#Moyenne des deux yeux


		if leftEAR < EYE_AR_THRESH:						#Si l'oeil gauche est considéré comme fermé
			COUNTER_left += 1							#Incrémentation du compteur de frame de fermeture de l'oeil
			OPEN_COUNTER_left = 0						
			if COUNTER_left >= EYE_AR_CONSEC_FRAMES:	#Si l'oei est fermé depuis "EYE_AR_CONSEC_FRAMES", alors affiché qu'il est fermé (permet de ne pas déclencher par erreur avec un clignement d'une frame
				OEIL=True								#Au moins un oeil fermé
				OEIL_gauche=True						#Oeil gauche fermé
		else:											#Sinon, si l'oeil gauche est ouvert
			OPEN_COUNTER_left += 1						#On compte le nombre de frame depuis qu'il s'est ouvert
			if OEIL == True and OPEN_COUNTER_left < EYE_CLOSE_FRAMES:	#Si l'oeil était fermé et qu'il n'est ouvert que depuis EYE_CLOSE_FRAMES frames, le considérer encore comme fermé
				OEIL=True								#L'oeil est encore considéré comme fermé
				OEIL_gauche=True

			else:										#Sinon, le considérer comme ouvert
				COUNTER_left = 0						#On réinitialise le compteur
				OEIL=False								
				OEIL_gauche=False

		#Même chose avec l'oeil droit
		if rightEAR < EYE_AR_THRESH:
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

		txt = str(compteur_frame)+"  "		#Affichage du nombre de frames dans la console
			
			
		if OEIL==True and mar > MOUTH_AR_THRESH and a1<40 and a1>-40:	#Si l'oeil est fermé et la bouche ouverte
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
