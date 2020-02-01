import librosa
import os
from random import shuffle
import numpy as np
from sklearn import svm
import sklearn


class vdisc():
    def __init__(self, path, label_num, feature_num):
        self.path = path
        self.label_num = label_num
        self.feature_num = feature_num

    def get_path(self):
        self.voice_path = []
        speaker_path = os.listdir(self.path)
        for speaker in speaker_path:
            temp1 = os.path.join(self.path, speaker)
            emotion_path = os.listdir(temp1)
            for i in range(len(emotion_path)):
                temp2 = os.path.join(temp1, emotion_path[i])
                voice_files = os.listdir(temp2)
                for voice in voice_files:
                    if voice.endswith('wav'):
                        self.voice_path.append([os.path.join(temp2, voice), i])
        shuffle(self.voice_path)

    def get_all_data(self):#please rewrite
        fst = 0
        self.all_feature = []
        self.all_label = []
        cnt = 0
        for voice in self.voice_path:
            y, sr = librosa.load(voice[0])
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=self.feature_num)
            zcr = librosa.feature.zero_crossing_rate(y)
            rmse = librosa.feature.rms(y)

            mfcc = np.mean(mfcc.T, axis=0).reshape(1, 13)
            zcr = np.mean(zcr.T, axis=0).reshape(1, 1)
            rmse = np.mean(rmse.T, axis=0).reshape(1, 1)

            if fst == 0:
                self.all_feature = np.concatenate((mfcc, zcr, rmse), axis=1)
                self.all_label=np.array([voice[1]]).reshape(1,1)
                fst = 1
            else:
                self.all_feature = np.concatenate((self.all_feature, np.concatenate((mfcc, zcr, rmse), axis=1)), axis=0)
                self.all_label = np.concatenate((self.all_label, np.array([voice[1]]).reshape(1,1)), axis=0)
            print(self.all_feature.shape)
            print(self.all_label.shape)
            cnt += 1
            if cnt % 100 == 0:
                print(str(cnt) + "/9595")

    def train_svm(self):
        split_num = 1600
        train_data = self.all_feature[:split_num, :]
        train_label = self.all_label[:split_num]
        test_data = self.all_feature[split_num:, :]
        test_label = self.all_label[split_num:]
        clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=19, gamma=0.0001)
        clf.fit(train_data, train_label)
        acc_train = sklearn.metrics.accuracy_score(clf.predict(train_data), train_label)
        acc_test = sklearn.metrics.accuracy_score(clf.predict(test_data), test_label)
        print('acc train ', acc_train)
        print('acc test ', acc_test)

    def predict(self, new_path):
        labelmap = {1: 'angry', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.new_path = new_path
        y, sr = librosa.load(self.new_path)
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=self.feature_num)
        zcr = librosa.feature.zero_crossing_rate(y)
        rmse = librosa.feature.rms(y)

        mfcc = np.mean(mfcc.T, axis=0)
        zcr = np.mean(zcr.T, axis=0)
        rmse = np.mean(rmse.T, axis=0)

        self.pre_feature = np.concatenate((mfcc, zcr, rmse), axis=1)
        self.pre_label = 1
        print(labelmap[self.pre_label])


def main():
    path = "E:/SortedData"
    # new_path = "E:/SortedData/my_voice.wav"
    label_num = 6
    feature_num = 13
    classifier = vdisc(path, label_num, feature_num)
    classifier.get_path()
    print("Completed")
    print("Num:" + str(len(classifier.voice_path)))
    classifier.get_all_data()
    print("Completed")
    classifier.train_svm()
    print("Completed")
    # classifier.predict(new_path)


if __name__ == "__main__":
    main()
