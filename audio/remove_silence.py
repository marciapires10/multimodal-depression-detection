import os

import librosa as librosa
import soundfile as sf
from scipy.io import wavfile


def remove_silence(participant_id):

    y, sr = librosa.load(f'wav/{participant_id}_AUDIO.wav')

    # plt.figure()
    # display.waveshow(y=y, sr=sr)
    # plt.xlabel("Time (seconds) ==>")
    # plt.ylabel("Amplitude")
    # plt.show()

    clips = librosa.effects.split(y, top_db=30)
    # print(clips)

    wav_data = []
    for c in clips:
        data = y[c[0]:c[1]]
        wav_data.extend(data)
    sf.write(f'wav_wosilence/{participant_id}_wosilence.wav', wav_data, sr)


for p in range(300, 493):
    try:
        remove_silence(p)
        print("Participant " + str(p))
    except:
        print("Participant " + str(p) + " doesn't exist.")


## noise reduction
# import noisereduce as nr
#
# rate, data = wavfile.read('301_wosilence.wav')
# reduced_noise = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=1.5, stationary=True)
# wavfile.write("301_reduced_noise.wav", rate, reduced_noise)