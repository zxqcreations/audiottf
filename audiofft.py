from pydub import AudioSegment
import wave
import pyaudio
import pylab
import numpy as np
import cv2
import time

# song = AudioSegment.from_mp3('e:\\matakimito.mp3')
# song.export('e:\\test.wav',format='wav')



wf  = wave.open('e:\\test.wav', 'rb')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=int(8820*2.5),#rate=wf.getframerate(),
                output=True)
nframes = wf.getnframes()
rate = wf.getframerate()
print(rate)
str_data = wf.readframes(nframes)
wf.close()
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = wave_data.reshape((-1, 2)).T
print(len(wave_data[0]))
#N = 1152*10
N = 44100
start = 0
df = rate/(N-1)
#df=1
freq = [df*n for n in range(0, 44100)]
t_size = 4000/5
img = np.zeros((int(0.75*t_size), int(t_size), 1))
tem_f = np.zeros((int(t_size)), dtype=np.int16)

#fourcc = cv2.VideoWriter_fourcc(*'mjpg')
#videoWriter = cv2.VideoWriter('e:\\test.avi', -1, 20, #*44100/11520,
#                              (int(t_size), int(t_size*0.75)))

tmp_wav = wave_data[0][0:8820]
stream.write(bytes(tmp_wav))
while True:
    f_cp = tem_f.copy()/2
    img.fill(0)
    #t1 = time.time()*10000
    #print([start, start+N])
    wave_data2 = wave_data[0][start:start+N]
    if bytes(wave_data2) == '' or len(wave_data2)==0:
        break
    c = np.fft.fft(wave_data2)*2/N
    c = abs(c)
    d = int(len(c)/2)
    while freq[d]>4000:
        d-=10
    
    #for j in range(0, 10):
    #    for i in range(1, d-1):
    #        if c[i]-c[i-1] < 0 and c[i]-c[i+1] < 0:
    #            c[i] = (c[i-1]+c[i+1])/2
    # tmp = np.zeros((1,d/50))
    ind = 0
    for i in range(0,d-10,10):
        tem_f[ind]=int(np.sum(c[i:i+10])/10)
        ind+=1
        

    # for i in range(0, len(c)):
    #     c[i] = tmp[i+1]

    #tmp = c.copy()
    #print(c.max())

    # img = np.zeros((2000, 4000, 1))
    h=int(0.75*t_size/2)
    '''
    for p in range(0, 21):
        img.fill(0)
        for i in range(0, len(tem_f)-1):
            if p<=10:
                cv2.line(img, (i, int(f_cp[i]+(-f_cp[i]+tem_f[i])*p/10)+h),
                         (i+1, int(f_cp[i+1]+(-f_cp[i+1]+tem_f[i+1])*p/10)+h),
                         (1, 1, 1))
            else:
                cv2.line(img, (i, int(tem_f[i]+(-tem_f[i]+tem_f[i]/2)*(p-10)/10)+h),
                         (i+1, int(tem_f[i+1]+(-tem_f[i+1]+tem_f[i+1]/2)*(p-10)/10)+h),
                         (1, 1, 1))
        #img = cv2.resize(img, (800, 400))
        cv2.imshow('tes', img)
        #videoWriter.write(img)
        #cv2.waitKey(5)
    '''
    for i in range(0, len(tem_f)-1):
        cv2.line(img, (i*2, int(-tem_f[i]/5)+h),
                 ((i+1)*2, int(-tem_f[i+1]/5)+h), (1, 1, 1))
    cv2.imshow('tes', img)
    r = int(8820*2)
    start += r
    stream.write(bytes(wave_data2[r:r+r]))
    #t2 = time.time()*10000
    #dt = t2-t1
    #if dt < 10000:
    #    time.sleep((10000-dt)/10000)

    if cv2.waitKey(10)>0:
        break

    # pylab.plot(freq[:d-1], c[:d-1])
    # pylab.show()
    # break

#videoWriter.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
