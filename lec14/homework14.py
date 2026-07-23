
!pip -q install gtts SpeechRecognition librosa soundfile pydub
!apt-get -qq install ffmpeg


from google.colab import files
uploaded = files.upload()


import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
from IPython.display import Audio


def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech,
    then write it to filename.
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files,
    convert MP3 -> WAV,
    recognize speech,
    and return recognized text.
    '''

    recognizer = sr.Recognizer()
    recognized_texts = []

    for text, lang, root in zip(texts, languages, filenames):

        mp3_file = root + ".mp3"
        wav_file = root + ".wav"

        # Step 1: Text -> MP3
        synthesize(text, lang, mp3_file)

        # Step 2: MP3 -> WAV
        audio, sample_rate = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, audio, sample_rate)

        # Step 3: Speech Recognition
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        try:
            recognized = recognizer.recognize_google(audio_data, language=lang)
        except:
            recognized = "Recognition Failed"

        recognized_texts.append(recognized)

    return recognized_texts

texts = [
    "This is speech synthesis!",
    "Artificial Intelligence is amazing.",
    "Hello everyone."
]

languages = [
    "en",
    "en",
    "en"
]

filenames = [
    "speech1",
    "speech2",
    "speech3"
]

results = make_a_corpus(texts, languages, filenames)

print("="*60)
print("Recognized Texts from Synthesized Speech")
print("="*60)

for i, text in enumerate(results, start=1):
    print(f"{i}. {text}")


print("\n" + "="*60)
print("Teacher WAV File")
print("="*60)

recognizer = sr.Recognizer()

with sr.AudioFile("lec14_speech.wav") as source:
    audio = recognizer.record(source)

try:
    wav_text = recognizer.recognize_google(audio, language="en")
    print(wav_text)
except:
    print("Recognition Failed")


print("\n" + "="*60)
print("Teacher MP3 File")
print("="*60)

audio_data, sample_rate = librosa.load("lec14_speech.mp3", sr=None)
sf.write("teacher.wav", audio_data, sample_rate)

with sr.AudioFile("teacher.wav") as source:
    audio = recognizer.record(source)

try:
    mp3_text = recognizer.recognize_google(audio, language="en")
    print(mp3_text)
except:
    print("Recognition Failed")


print("\n" + "="*60)
print("lec14_speech WAV")
print("="*60)
display(Audio("lec14_speech.wav"))

print("="*60)
print("lec14_speech MP3")
print("="*60)
display(Audio("lec14_speech.mp3"))

print("="*60)
print("Play Synthesized Speech")
print("="*60)
display(Audio("speech1.mp3"))
