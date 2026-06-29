import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.

    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio

    @returns:
    text (str) - the recognized speech
    '''

    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio, language=language)

    return text


if __name__ == "__main__":
    filename = "speech_waveform.wav"
    language = "en"

    try:
        text = transcribe_wavefile(filename, language)
        print("Recognized Text:")
        print(text)
    except Exception as e:
        print("Error:", e)
