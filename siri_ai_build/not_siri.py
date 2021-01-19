## build a siri bot that responds to questions
# takes in live chat and responds with information
import speech_recognition as sr

listener = sr.Recognizer()
try:
    with sr.Microphone() as source:
        print("go ahead, im listening")
        voice = listener.listen(source)
        command = listener.recognize_google(voice)
        print(command)

except:
    pass
