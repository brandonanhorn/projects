import random
import time

import speech_recognition as sr

def recognize_speech_from_mic(recognizer, microphone):
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    return response

if __name__ == "__main__":
    words = ["blue", "orange", "yellow", "purple", "pink", "green"]
    num_of_guesses = 3
    prompt_limit = 5

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    word = random.choice(words)

    instructions = (
        "I'm thinking of one of these colors:\n"
        "{words}\n"
        "You have {n} tries to guess which one.\n"
    ).format(words=', '.join(words), n=num_of_guesses)

    print(instructions)
    time.sleep(3)

    for i in range(num_of_guesses):
        for j in range(prompt_limit):
            print('Guess {}. Speak!'.format(i+1))
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("I didn't catch that. Try again.\n")

        if guess["error"]:
            print("ERROR: {}".format(guess["error"]))
            break

        print("You said: '{}'".format(guess["transcription"]))

        guess_is_correct = guess["transcription"].lower() == word.lower()
        user_has_more_attempts = i < num_of_guesses - 1

        if guess_is_correct:
            print("Correct! You win!".format(word))
            break
        elif user_has_more_attempts:
            print("Incorrect. Try again.\n")
        else:
            print("Sorry, you lose!\nI was thinking of '{}'.".format(word))
            break
