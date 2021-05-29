import numpy as np
import torch
import torch.nn as nn
import torchaudio as audio
import sklearn
import speech_recognition as sr
import pyaudio
import threading
import wave

def speech_recog():

    mic = sr.Microphone()
    with mic as audio_file:
        print("Speak Please")

        recog = sr.Recognizer()

        recog.adjust_for_ambient_noise(audio_file)
        audio = recog.listen(audio_file, phrase_time_limit=4)

    try:
        phrase = recog.recognize_google(audio, language="ru-RU")
    except:
        print('try again')
        phrase = speech_recog()

    print("Converting Speech to Text...")
    print("You said: " + phrase)

    return phrase

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
RATE = 16000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

window_time = 0.01 # seconds
window_size = int(RATE * window_time)

spectogram = audio.transforms.Spectrogram(
    n_fft=window_size,
    win_length=window_size,
    hop_length=window_size // 2,
    power=2
)

print("* recording")

frames = []
h = torch.zeros(1, 32)
recog = sr.Recognizer()

while True:
    data = stream.read(CHUNK)
    audio = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32))
    audio_spec = spectogram(audio)

    net.eval().cpu()
    with torch.no_grad():
        for i, x in list(enumerate(audio_spec[:40, :].T)):
            pred_word = 0
            prob = 0
            pred, h = net(x.view(1, -1), h)
            if torch.exp(pred[0][1]).item() > 0.95:
                pred_word = 1
                print(i, 'ЭТО БЫЛА АЛИНА')
                speech_recog()

    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()



