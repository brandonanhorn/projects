import googletrans
import speech_recognition as sr
import gtts
import playsound

#print(googletrans.LANGUAGES)
recognizer = sr.Recognizer()
translator = googletrans.Translator()
input_lang = "bs"
output_lang = "en"

try:
    with sr.Microphone() as source:
        print("Talk now")
        voice = recognizer.listen(source)
        text = recognizer.recognize_google(voice, language=input_lang)
        print(text)
except:
    pass

translated = translator.translate(text, dest=output_lang)
print(translated.text)
converted_audio = gtts.gTTS(translated.text, lang=output_lang)
converted_audio.save("otherlang.mp3")
playsound.playsound("otherlang.mp3")
