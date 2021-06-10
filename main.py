import numpy as np
import torch
import torch.nn as nn
import librosa
import speech_recognition as sr
import pyaudio
from gtts import gTTS
import os
import threading
import wave
import re

class NoexceptStream:
    def __init__(self, stream):
        self.stream = stream
    def read(self, *args):
        return self.stream.read(*args, exception_on_overflow=False) 

class ReusableMicrophone(sr.AudioSource):
    def __init__(self, stream):
        self.stream = NoexceptStream(stream)
        self.CHUNK = CHUNK
        self.SAMPLE_WIDTH = 2
        self.SAMPLE_RATE = RATE
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def speech_recog(stream):
    myobj = gTTS(text='говорите', lang='ru', slow=False).save("wakeup_reaction.mp3")
    os.system("mpg321 wakeup_reaction.mp3")

    # mic = sr.Microphone()
    mic = ReusableMicrophone(stream)
    with mic as audio_file:
        print("Speak Please")

        recog = sr.Recognizer()

        recog.adjust_for_ambient_noise(audio_file)
        audio = recog.listen(audio_file, phrase_time_limit=4)

    try:
        phrase = recog.recognize_google(audio, language="ru-RU")
    except:
        print('try again')
        phrase = 'error'

    print("You said: " + phrase)

    #return phrase

class AlinaRNN(nn.Module):
    def __init__(self, code_size, hidden_size):
        super(AlinaRNN, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(code_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.rnn = nn.GRUCell(128, hidden_size)
        self.decode = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1),
        )
    def forward(self, x, h):
        h = self.rnn(self.encode(x), h)
        return self.decode(h), h


net = AlinaRNN(40, 32).float()
net.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 2

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = []
window_time = 0.01 # seconds
window_size = int(RATE * window_time) // 3

spectrogram = lambda x: torch.tensor(librosa.stft(x.numpy(), n_fft=window_size, win_length=window_size, hop_length=window_size//2)).abs() ** 2


print("* recording")

frames = []
h = torch.zeros(1, 32)
recog = sr.Recognizer()


j = 0

while True:
    data = stream.read(CHUNK * 3, exception_on_overflow = False)
    audio = torch.from_numpy(np.frombuffer(data, dtype=np.int16)[::3].astype(np.float32))
    audio_spec = spectrogram(audio)
    frames.append(data)
    if len(frames) > 50:
        frames.pop(0)

    net.eval().cpu()
    with torch.no_grad():
        flag = True
        for i, x in list(enumerate(audio_spec[:40, :].T)):
            prob = 0
            pred, h = net(x.view(1, -1), h)
            if flag and torch.exp(pred[0][1]).item() > 0.95:
                print('Alina?')                

                wf = wave.open('check.wav', 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                AUDIO_FILE = 'check.wav'
                r = sr.Recognizer()
                with sr.AudioFile(AUDIO_FILE) as source:
                    check = r.record(source)
                    try:
                        phrase = r.recognize_google(check, language="ru-RU")
                    except:
                        phrase = ''
                    print(phrase)
                    if not len(re.findall('Алина', phrase)):
                        continue
                flag = False
                j = 0
                speech_recog(stream)

    j += 1

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
