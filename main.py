import numpy as np
import argparse
import time
import glob
from textblob import TextBlob
import goslate
from playsound import playsound
from PIL import Image
import pytesseract
import cv2
import os
from textblob import TextBlob
from gtts import gTTS
import speech_recognition as sr 
import pyttsx3  
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
weightsPath = "yolov3-tiny.weights"
configPath ="yolov3-tiny.cfg"
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
r = sr.Recognizer()  
def SpeakText(command):  
	engine = pyttsx3.init() 
	engine.setProperty('rate', 150) 
	engine.say(command) 
	engine.runAndWait() 

SpeakText("Please Say Your Query for read more say one and for blind mode say two")
def lis():
	s=True
	while(s):	  
			try: 
				with sr.Microphone() as source2: 
						r.adjust_for_ambient_noise(source2, duration=0.2)  
						audio2 = r.listen(source2) 
						MyText = r.recognize_google(audio2) 
						MyText = MyText.lower() 
						print("Did you say "+MyText) 
						s=False
						SpeakText(MyText) 
						return str(MyText)
			except sr.RequestError as e: 
				print("Could not request results; {0}".format(e)) 
			except sr.UnknownValueError: 
					print("unknown error occured") 
	return MyText
out=lis()
print(out)
for img in glob.glob("C:\\Users\\tanka\\object detection\\t1.png"):
	if out=="two" or out=="2":
				image = cv2.imread(img)
				(H, W) = image.shape[:2]

				# determine only the *output* layer names that we need from YOLO
				ln = net.getLayerNames()
				ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
				blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
					swapRB=True, crop=False)
				net.setInput(blob)
				start = time.time()
				layerOutputs = net.forward(ln)
				end = time.time()

				# show timing information on YOLO
				print("[INFO] YOLO took {:.6f} seconds".format(end - start))

				# initialize our lists of detected bounding boxes, confidences, and
				# class IDs, respectively
				boxes = []
				confidences = []
				classIDs = []

				for output in layerOutputs:
					for detection in output:
						scores = detection[5:]
						classID = np.argmax(scores)
						confidence = scores[classID]
						if confidence > 0.3:
							box = detection[0:4] * np.array([W, H, W, H])
							(centerX, centerY, width, height) = box.astype("int")
							x = int(centerX - (width / 2))
							y = int(centerY - (height / 2))
							boxes.append([x, y, int(width), int(height)])
							confidences.append(float(confidence))
							classIDs.append(classID)
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3,0.3)
				
				lab=[]
				conf=[]
				bb=[]
				e={}
				if len(idxs) > 0:
					for i in idxs.flatten():
						(x, y) = (boxes[i][0], boxes[i][1])
						(w, h) = (boxes[i][2], boxes[i][3])
						color = [int(c) for c in COLORS[classIDs[i]]]
						cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
						text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
						cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
							0.5, color, 2)
						bb.append(w*h)#AREA
						lab.append(LABELS[classIDs[i]])#LABELS
						conf.append(confidences[i])#PROBA
						dictionary = dict(zip(bb, zip(lab, conf)))
						dic = dict(zip(conf,lab))
						from collections import Counter
						c = Counter(dic.values())
				#print(c)
				a="There are"
				#print(c.items())
				for i,j in c.items():
					a=a+" "+str(j)+" "+i+" "
				a=(a+"in front of you")
				t=gTTS(str(a),lang="en")
				t.save("h.mp3")
				print(a)
				SpeakText(a)
				cv2.imshow("Image", image)
				cv2.waitKey(0)
	if out=="one" or out=="1":
		SpeakText("which language Do You Want for english say one for Telugu say two for hindi say three for tamil say four")
		out1=lis()
		out1="three"
		if out1=="one" or out1=="1":
			out1="English"
		elif out1=="tu" or out1=="2":
			out1="Telugu" 
		elif out1=="three" or out1=="3":
			out1="Hindi" 
		elif out1=="four" or out1=="4":
			out1="Tamil" 
		print(out1)
		dic={'Latin': 'la', 'Filipino': 'tl', 'Spanish': 'es', 'Russian': 'ru', 'Swahili': 'sw', 'Hungarian': 'hu', 'Gujarati' : 'gu', 'Myanmar (Burmese)': 'my', 'Telugu': 'te', 'Sinhala': 'si', 'Albanian': 'sq', 'Marathi': 'mr', 'Dutch': 'n l', 'Bengali': 'bn', 'Vietnamese': 'vi', 'Korean': 'ko', 'Kannada': 'kn', 'Turkish': 'tr', 'Czech': 'cs', 'Croatia n': 'hr', 'Icelandic': 'is', 'German': 'de', 'Welsh': 'cy', 'Estonian': 'et', 'Thai': 'th', 'Nepali': 'ne', 'Frenc h': 'fr', 'Danish': 'da', 'Portuguese': 'pt', 'Japanese': 'ja', 'Norwegian': 'no', 'Armenian': 'hy', 'Catalan': 'c a', 'Romanian': 'ro', 'Indonesian': 'id', 'Swedish': 'sv', 'Malayalam': 'ml', 'Hindi': 'hi', 'Arabic': 'ar', 'Serb ian': 'sr', 'Macedonian': 'mk', 'Khmer': 'km', 'Sundanese': 'su', 'Javanese': 'jw', 'Bosnian': 'bs', 'Greek': 'el' , 'Tamil': 'ta', 'Finnish': 'fi', 'Urdu': 'ur', 'Chinese': 'zh-CN', 'English': 'en', 'Polish': 'pl', 'Italian': 'i t', 'Esperanto': 'eo', 'Slovak': 'sk', 'Afrikaans': 'af', 'Ukrainian': 'uk', 'Latvian': 'lv'}
		if out1 in list(dic.keys()):
			lan=dic[out1]
		else:
			break
		image = cv2.imread(img)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		gray = cv2.medianBlur(gray, 3)
		filename = "{}.png".format(os.getpid())
		cv2.imwrite(filename, gray)
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
		text = pytesseract.image_to_string(Image.open(filename))
		os.remove(filename)
		summary = TextBlob(text)
		print(summary,lan)
		if lan!='en':
			summary= summary.translate(from_lang='en',to=lan)
			print(summary)
		t=gTTS(str(summary),lang=lan)
		t.save("audio.mp3")
		playsound('audio.mp3')
		cv2.imshow("Image", image)
		cv2.imshow("Output", gray)
		cv2.waitKey(0)