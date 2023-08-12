import numpy as np
import os
from scipy.io import wavfile
from matplotlib import pyplot as plt


content = os.listdir(r'C:\Users\Dell\Desktop\audio\Original_audios')
count = 1

for j in range(len(content)):
    fs, data = wavfile.read(r'C:\Users\Dell\Desktop\audio\Original_audios\\' + content[j])
    if len(data.shape) == 2:
        norm = max(abs(data[:, 0]))
        data = data[:, 0]/norm
    else:
        norm = max(abs(data))
        data = data/norm
    #plt.axis('off')
    plt.hist(data, bins = 256, color = "black")
    plt.savefig(r'C:\Users\Dell\Desktop\audio\Original_audios_hist\test' + str(count) + '.jpg')
    count += 1
    plt.clf()


content = os.listdir(r'C:\Users\Dell\Desktop\audio\Fake_audios')
count = 1

for j in range(len(content)):
    fs, data = wavfile.read(r'C:\Users\Dell\Desktop\audio\Fake_audios\\' + content[j])
    if len(data.shape) == 2:
        norm = max(abs(data[:, 0]))
        data = data[:, 0]/norm
    else:
        norm = max(abs(data))
        data = data/norm
    #plt.axis('off')
    plt.hist(data, bins = 256, color = "black")
    plt.savefig(r'C:\Users\Dell\Desktop\audio\Fake_audios_hist\test' + str(count) + '.jpg')
    count += 1
    plt.clf()
