from pydub import AudioSegment
import wave
import pyaudio
import pylab
import numpy as np
import cv2
import time
import subprocess
from scipy.interpolate import spline

# song = AudioSegment.from_mp3('e:\\test1.mp3')
# song.export('e:\\test1.wav',format='wav')

#raise IOError('123')

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
df = rate/(N-1)
inter = 10
r = int(N/inter)
#df=1
freq = [df*n for n in range(0, rate)]
t_size = 1366
t_size_h = t_size*9/16
c_d = int(t_size_h*0.5)
img = np.zeros((int(t_size_h), int(t_size), 3))
img_s = cv2.imread('e:\\bg.jpg')
img_s = cv2.resize(img_s, (int(t_size), int(t_size_h)))
fft_result = list()

from_color = (33, 150, 243)
end_color = (76, 175, 80)


#fourcc = cv2.VideoWriter_fourcc(*'mjpg')
videoWriter = cv2.VideoWriter('e:\\test.avi', -1, inter, #*44100/11520,
                              (int(t_size), int(t_size_h)))
print('Strating FFT')
t1 = time.time()*10
while True:
    tem_f = list()
    wave_data2 = wave_data[0][start:start+N]
    if bytes(wave_data2) == '' or len(wave_data2)==0:
        break
    c = abs(np.fft.fft(wave_data2)*2/N)
    d = int(len(c)/2)
    while freq[d]>3000:
        d-=10
    for i in range(0,d-inter,inter):
        mf = int(np.sum(c[i:i+inter])/inter)
        if mf > 1000:
            mf = 1000
        tem_f.append(mf)
    #x_o = np.linspace(0,len(tem_f)-1, len(tem_f))
    #x = np.linspace(0, len(tem_f)-1, t_size-1)
    #y = spline(x_o, np.asarray(tem_f), x)
    fft_result.append(tem_f)
    start += r
    #print(start)
    #print('processing, now {:0.1f}%\r'.format(start*100/len(wave_data[0])))
t2 = time.time()*10
print('\nFFT finished, generating, took {:0.1f} s'.format((t2-t1)/10))

start = 0
tmp_wav = wave_data[0][0:int(inter/2)*r]
stream.write(bytes(tmp_wav))
for fft in fft_result:
    img = img_s.copy()
    #print([start, start+N])
    wave_data2 = wave_data[0][start:start+N]
    h=int(t_size_h/2)
    pt_t=(0,0,0,0)
    for i in range(0, len(fft)-1):
        degree = 1/len(fft)
        x1 = int((c_d/2+fft[i]/10)*np.cos(degree*i*2*np.pi-np.pi/2) + t_size/2)
        y1 = int((c_d/2+fft[i]/10)*np.sin(degree*i*2*np.pi-np.pi/2) + h)
        x2 = int((c_d/2+fft[i+1]/10)*np.cos(degree*(i+1)*2*np.pi-np.pi/2) + t_size/2)
        y2 = int((c_d/2+fft[i+1]/10)*np.sin(degree*(i+1)*2*np.pi-np.pi/2) + h)
        if i==0:
            pt_t=(x1,y1)
        #(33, 150, 243), (76, 175, 80)
        if degree*i <= 0.5:
            cv2.line(img, (x1, y1), (x2, y2),
                     (from_color[2]+(end_color[2]-from_color[2])*degree*i*2,
                      from_color[1]+(end_color[1]-from_color[1])*degree*i*2,
                      from_color[0]+(end_color[0]-from_color[0])*degree*i*2), 3)
        else:
            cv2.line(img, (x1, y1), (x2, y2),
                     (end_color[2]+(from_color[2]-end_color[2])*(degree*i-0.5)*2,
                      end_color[1]+(from_color[1]-end_color[1])*(degree*i-0.5)*2,
                      end_color[0]+(from_color[0]-end_color[0])*(degree*i-0.5)*2), 3)
        if i==len(fft)-2:
            cv2.line(img, (x2, y2), (pt_t[0], pt_t[1]),
                     (end_color[2]+(from_color[2]-end_color[2])*(degree*i-0.5)*2,
                      end_color[1]+(from_color[1]-end_color[1])*(degree*i-0.5)*2,
                      end_color[0]+(from_color[0]-end_color[0])*(degree*i-0.5)*2), 3)
    cv2.imshow('tes', img)
    videoWriter.write(img)
    start += r
    stream.write(bytes(wave_data2[int(inter/2)*r:(int(inter/2)+1)*r]))
    if cv2.waitKey(10)>0:
        break

videoWriter.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()

#subprocess.call('ffmpeg -i e:\\test.avi -i e:\\test.mp3 -strict -2 -f avi e:\\test_ok.avi',
#                shell=True)
