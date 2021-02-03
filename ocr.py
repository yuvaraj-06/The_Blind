from PIL import Image
import pytesseract
import cv2
import os
import speech_recognition as sr 
import pyttsx3 
r = sr.Recognizer()  
def SpeakText(command):  
	engine = pyttsx3.init() 
	engine.say(command) 
	engine.runAndWait() 
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
        except sr.RequestError as e: 
            print("Could not request results; {0}".format(e)) 
        except sr.UnknownValueError: 
                print("unknown error occured") 
if MyText=="read mode":
    image = cv2.imread("img.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print(text)
    cv2.imshow("Image", image)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)