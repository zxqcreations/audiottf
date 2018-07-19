from pydub import AudioSegment
import wave
import pyaudio
import pylab
import numpy as np
import cv2
import time


wf  = wave.open('e:\\test.wav', 'rb')
nframes = wf.getnframes()
rate = wf.getframerate()
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=int(rate/2),#rate=wf.getframerate(),
                output=True)
print(rate)
str_data = wf.readframes(nframes)
wf.close()
wave_data = np.fromstring(str_data, dtype=np.short).reshape((-1, 2)).T
print(len(wave_data[0]))
N = rate
start = 0

while True:
    wave_data2 = wave_data[0][start:start+N]
    if bytes(wave_data2) == '' or len(wave_data2)==0:
        break
    stream.write(bytes(wave_data2))
    start+=int(N)

stream.stop_stream()
stream.close()
p.terminate()
