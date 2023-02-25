import os

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import nltk
import numpy as np
import prosodic as p
from essential_generators import DocumentGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from keras.models import Model, load_model
from keras.utils import np_utils
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numba import cuda, jit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

OUTPUT_MODEL_FILE = 'trained.hdf5'
OUTPUT_CLASSES_FILE = 'trined.txt'
OUTPUT_ACC_PLOT = 'acc_train_plot.jpg'
SAMPLE_RATE = 8000

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')


class Audio:
    def __init__(self, audio_file: str, figsize=(14, 5)):
        self.audio_file = audio_file
        self.x, self.sr = librosa.load(audio_file)
        self.figsize = figsize

    def test(self):
        print('Testing')

    def waveshow(self):
        plt.figure(figsize=self.figsize)
        librosa.display.waveshow(self.x, sr=self.sr)

    def spectogram(self, log=False):
        X = librosa.stft(self.x)
        Xdb = librosa.amplitude_to_db(abs(X))
        y = 'log' if log else 'hz'
        plt.figure(figsize=self.figsize)
        plt.title(f'Spektogram, typ:{y}')
        librosa.display.specshow(
            Xdb, sr=self.sr, x_axis='time', y_axis=y)
        plt.colorbar()

    def play(self):
        ipd.Audio(self.audio_file)

    def write_to_file(self, filename: str):
        librosa.output.write_wav(filename, self.x, self.sr)

    def mfcc(self):
        mfcc = librosa.feature.mfcc(y=self.x, sr=self.sr)
        plt.figure(figsize=self.figsize)
        librosa.display.specshow(mfcc, sr=self.sr, x_axis='time')
        plt.show()


class SpeechToText():
    def __init__(self, train_audio_path: str, output_model_file: str, sample_rate=SAMPLE_RATE, figsize=(14, 5)):
        self.train_audio_path = train_audio_path
        self.figsize = figsize
        self.sample_rate = sample_rate
        self.output_model_file = output_model_file

    # @jit(target_backend='cuda')
    def train(self):
        all_wave = []
        all_label = []
        labels = os.listdir(self.train_audio_path)
        for label in labels:
            print(label)
            waves = [f for f in os.listdir(
                self.train_audio_path + '/' + label) if f.endswith('.wav')]
            for wav in waves:
                samples, sample_rate = librosa.load(
                    self.train_audio_path + '/' + label + '/' + wav, sr=16000)
                samples = librosa.resample(
                    samples, sample_rate, self.sample_rate)
                if (len(samples) == self.sample_rate):
                    all_wave.append(samples)
                    all_label.append(label)
        #
        le = LabelEncoder()
        y = le.fit_transform(all_label)
        classes = list(le.classes_)

        with open(OUTPUT_CLASSES_FILE, 'w') as fp:
            fp.write('\n'.join(classes))
            fp.write('\n')

        #
        y = np_utils.to_categorical(y, num_classes=len(labels))

        all_waves = np.array(all_wave).reshape(-1, self.sample_rate, 1)

        # split to  train and test

        TEST_SIZE = 0.2
        x_tr, x_val, y_tr, y_val = train_test_split(
            np.array(all_wave), np.array(y), stratify=y, test_size=TEST_SIZE, random_state=7)

        # conv layers
        K.clear_session()
        inputs = Input(shape=(self.sample_rate, 1))
        # First Conv1D layer
        conv = Conv1D(8, 13, padding='valid',
                      activation='relu', strides=1)(inputs)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.3)(conv)
        # Second Conv1D layer
        conv = Conv1D(16, 11, padding='valid',
                      activation='relu', strides=1)(conv)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.3)(conv)
        # Third Conv1D layer
        conv = Conv1D(32, 9, padding='valid',
                      activation='relu', strides=1)(conv)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.3)(conv)
        # Fourth Conv1D layer
        conv = Conv1D(64, 7, padding='valid',
                      activation='relu', strides=1)(conv)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.3)(conv)
        # Flatten layer
        conv = Flatten()(conv)
        # Dense Layer 1
        conv = Dense(256, activation='relu')(conv)
        conv = Dropout(0.3)(conv)
        # Dense Layer 2
        conv = Dense(128, activation='relu')(conv)
        conv = Dropout(0.3)(conv)
        outputs = Dense(len(labels), activation='softmax')(conv)
        model = Model(inputs, outputs)
        model.summary()

        #
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=10, min_delta=0.0001)
        mc = ModelCheckpoint(
            self.output_model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        history = model.fit(x_tr, y_tr, epochs=100, callbacks=[
            es, mc], batch_size=32, validation_data=(x_val, y_val))

        SpeechToText.history(history)
        model.save(self.output_model_file)

    @staticmethod
    def history(history):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig(OUTPUT_ACC_PLOT)
        plt.show()

    @staticmethod
    def predict(audio_path) -> str:
        ss, sample_rate = librosa.load(audio_path, sr=16000)
        ss = librosa.resample(ss, sample_rate, SAMPLE_RATE)
        model = load_model(OUTPUT_MODEL_FILE)
        try:
            prob = model.predict(ss.reshape(1, SAMPLE_RATE, 1))
        except:
            return ()

        index = np.argmax(prob[0])

        print("Index    ")
        print(index)

        classes = []
        with open(OUTPUT_CLASSES_FILE, 'r') as fp:
            for line in fp:
                x = line[:-1]
                classes.append(x)

        return classes[index], index


class TextToSpeech():
    # TODO FIXME TEST
    def __init__(self):
        self.text = TextToSpeech.__rand_sentence()
        # self.text = 'Speech I like to'

    @staticmethod
    def __rand_sentence():
        gen = DocumentGenerator()
        return gen.gen_sentence(15, 60)

    def example(self):
        print(f'start_text:{self.text}')

        # tokenize
#         s = '''Good muffins cost $3.88\nin New York.  Please buy me
# ... two of them.\n\nThanks.'''
        tokenized = word_tokenize(self.text)
        print('edr')
        print(f'tokenized:{tokenized}')

        # freq``
        freq = FreqDist(tokenized)
        freq_ten = freq.most_common(10)
        print(f'freq10:{freq_ten}')

        # stemming
        ps = PorterStemmer()
        common = 'common_part'
        s_dict = []
        for i in range(33):
            end = i*'a'
            s_dict.append(common+end)

        print(f'Dict{s_dict}')
        for i in s_dict:
            print(f'stemmed:{i}:{ps.stem(i)}')

        # stop words
        stopped = set(stopwords.words())
        print(stopped)
        lokomotywa = 'A jeszcze palacz węgiel w nią sypie.Wagony do niej podoczepialiWielkie i ciężkie, z żelaza, stali,I pełno ludzi w każdym wagonie,A wjednym krowy, a w drugim konie,A w trzecim siedzą same grubasy,Siedzą i jedzą tłuste kiełbasy,A czwarty wagon pełen bananów,A w piątym stoi sześć fortepianów,W szóstym armata - o! jaka wielka!'
        lokomotywa = word_tokenize(lokomotywa.lower())
        my_stopwords = [i for i in lokomotywa if i not in stopped]
        print(f'Stopwords lokomotywa:{my_stopwords}')

        # prosodic: text -> rhythm text
        prosodic_sentence = 'Sometimes life hits you in the head with a brick. Do not lose faith.'
        print(f'prosodic sentence:{prosodic_sentence}')
        prosodic = p.Text(prosodic_sentence)
        prosodic_parsed = prosodic.parse()
        print(f'prosodic_parsed:{prosodic_parsed}')