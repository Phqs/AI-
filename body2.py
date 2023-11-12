from moviepy.editor import VideoFileClip, concatenate_videoclips
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
# Загрузка дата-сета

s = '/Users/User/Documents/ programming/projects/AI/audio_dataset/audio_files'
mpfile = [file for file in os.listdir(s) if os.path.isfile(os.path.join(s, file))]
csv = "/Users/User/Documents/programming/projects/AI/audio_dataset/df.csv"
#dataset_ = pd.read_csv(csv)

# = '/Users/../Documents/ programming/projects/AI/audio_dataset/audio_files/'
wavfile = '/Users/User/Documents/programming/projects/AI/audiotrackconverted/'
print(len(mpfile))

#Конвертауия файлов  mp3 в wav
def convert_to_wav(mp4_file, wav_file):
    video = VideoFileClip(mp4_file)
    audio = video.audio
    audio.write_audiofile(wav_file)

convert_to_wav(mpfile, wavfile)


#a = open('/Users/../Documents/ programming/projects/AI/audiotrackconverted/1a.wav')

# Загрузка аудио файла
audio_path = '/Users/User/Documents/ programming/projects/AI/audio_dataset/audio_files/'
audioData = [file for file in os.listdir(audio_path) if file.endswith('.wav') ]


def download_audio(file):
    path = os.path.join(audio_path, file)
    sound, sampling_frequency = librosa.load(path)
    return sound, sampling_frequency 

with ThreadPoolExecutor() as executor:
    result = list(executor.map(download_audio, audioData))

y, sr = librosa.load(result)
# Создание спектрограммы

D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

# Отображение спектрограммы
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Спектральная карта аудио файла')
plt.savefig('/Users/User/Documents/ programming/projects/AI/spectogramms/spectr.png')
plt.show()

# модель нейронной сети

from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Embedding
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(None, 1000)))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size = 33, activation='softmax')))

# архитектура сверточной нейронной сети

#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(input_shape= ((250, 250), 1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, input_dim=100, activation='relu'))
model.add(Dense(num_classes=2, activation='softmax'))

# Компиляция и обучение нейронной сети
from sklearn.model_selection import train_test_split

audio_data = '/Users/User/Documents/ programming/projects/AI/spectogramms'

X_train, X_temp, y_train, y_temp = train_test_split(98, 98, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Предсказание

predicted_text = model.predict()
new_audio_data = predicted_text
# Как выглядит нейросеть

model.summary()

from googletrans import Translator

translator = Translator()
lang_text = translator.translate(new_audio_data, src='ru', dest='en').textx