from random import shuffle
import numpy as np
import os
import librosa
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DataError(Exception):
    def __init__(self, info):
        super().__init__(self)
        self.info = info

    def __str__(self):
        return self.info


class svm_disc():
    def __init__(self):
        self.all_feature = None
        self.all_label = None
        self.train_feature = None
        self.train_label = None
        self.test_feature = None
        self.test_label = None
        self.train_acc = None
        self.test_acc = None
        self.classifier = None

    def allocate_data(self, train_rate=0.88):
        self.train_feature, self.test_feature, self.train_label, self.test_label = train_test_split(self.all_feature,
                                                                                                    self.all_label,
                                                                                                    test_size=1 - train_rate
                                                                                                    )

    def get_path(self, path):
        voice_path = []
        speaker_path = os.listdir(path)
        for speaker in speaker_path:
            temp1 = os.path.join(path, speaker)
            emotion_path = os.listdir(temp1)
            for i in range(len(emotion_path)):
                temp2 = os.path.join(temp1, emotion_path[i])
                voice_files = os.listdir(temp2)
                for voice in voice_files:
                    if voice.endswith('wav'):
                        voice_path.append([os.path.join(temp2, voice), i])
        shuffle(voice_path)
        return voice_path

    def get_all_data_p(self, path, feature_num):
        fst = 0
        voice_path = self.get_path(path)
        for voice in tqdm(voice_path, desc="Getting features: "):
            y, sr = librosa.load(voice[0])
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=feature_num)
            zcr = librosa.feature.zero_crossing_rate(y)
            rmse = librosa.feature.rms(y)

            mfcc = np.mean(mfcc.T, axis=0).reshape(1, feature_num)
            zcr = np.mean(zcr.T, axis=0).reshape(1, 1)
            rmse = np.mean(rmse.T, axis=0).reshape(1, 1)

            if fst == 0:
                self.all_feature = np.concatenate((mfcc, zcr, rmse), axis=1)
                self.all_label = np.array([voice[1]]).reshape(1, 1)
                fst = 1
            else:
                self.all_feature = np.concatenate((self.all_feature, np.concatenate((mfcc, zcr, rmse), axis=1)), axis=0)
                self.all_label = np.concatenate((self.all_label, np.array([voice[1]]).reshape(1, 1)), axis=0)
        if self.all_feature.shape == (9595, feature_num + 2) and self.all_label.shape == (9595, 1):
            print("Data shape：")
            print(self.all_feature.shape)
            print(self.all_label.shape)
            np.save("af.npy", self.all_feature)
            np.save("al.npy", self.all_label)
            self.all_label = self.all_label.reshape(9595, )
        else:
            raise DataError("Data shape unmatched!")

    def get_all_data_np(self, path1, path2):
        self.all_feature = np.load(path1)
        self.all_label = np.load(path2).reshape(9595, )
        if self.all_feature.shape == (9595, 22) and self.all_label.shape == (9595,):
            print("Data shape：")
            print(self.all_feature.shape)
            print(self.all_label.shape)
        else:
            raise DataError("Data shape unmatched!")

    def train_svm(self):
        self.classifier = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=15, gamma=0.0001)
        self.classifier.fit(self.train_feature, self.train_label)

    def svm_predict(self, test_sample):
        return self.classifier.predict(test_sample)

    def acc(self):
        self.train_acc = accuracy_score(self.svm_predict(self.train_feature), self.train_label)
        self.test_acc = accuracy_score(self.svm_predict(self.test_feature), self.test_label)
        print("train acc", self.train_acc)
        print("test acc", self.test_acc)


def main():
    path1 = r"af.npy"
    path2 = r"al.npy"
    # path = r"E:\SortedData"
    my_svm = svm_disc()
    my_svm.get_all_data_np(path1, path2)
    # my_svm.get_all_data_p(path, 20)
    print("Got")
    my_svm.allocate_data()
    print("Allocated")
    my_svm.train_svm()
    print("Built")
    my_svm.acc()


if __name__ == "__main__":
    main()
