## Authenticating & Setup
from ibm_watson import SpeechToTextV1, LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

language_translator_api = "pWGwiJ7I0huXzEs0X9CYJ8luUlCf5tzD6d3vKKOQpPmz"
language_translator_url = "https://api.us-south.language-translator.watson.cloud.ibm.com/instances/cdd24e6e-5f46-4c70-baf7-af865673369d"
speech_to_text_api = "CRhndbyi9ypc5Wu6ag8bW3ISP-qoE1xkGXqPnDqm0WsB"
speech_to_text_url = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/9bdd460d-b93a-4f37-a06c-369d0246b23f"

language_translator_authenticator = IAMAuthenticator(language_translator_api)
languaage_translator = LanguageTranslatorV3(version="2018-05-01", authenticator=language_translator_authenticator)
languaage_translator.set_service_url(language_translator_url)

speech_to_text_authenticator = IAMAuthenticator(speech_to_text_api)
speech_to_text = SpeechToTextV1(authenticator=speech_to_text_authenticator)
speech_to_text.set_service_url(speech_to_text_url)

## Speech to text
with open("Untitled.mp3", "rb") as f:
    a_file = speech_to_text.recognize(audio=f, content_type="audio/mp3",model="en-AU_NarrowbandModel", continuous=True).get_result()

a_file
voice_text = a_file["results"][0]["alternatives"][0]["transcript"]

## Text translation
french = "en-fr"

translation = languaage_translator.translate(text=voice_text, model_id=french).get_result()
translation_text = translation["translations"][0]["translation"]

## Create file from translation
with open("translation_text.txt", "w") as f:
    f.write(translation_text)
