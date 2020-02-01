import random
from itertools import combinations
from random import shuffle
import numpy as np
import os
import librosa
import json
import my_decision_tree
import copy
import time
from tqdm import tqdm


class DataError(Exception):
    def __init__(self, info):
        super().__init__(self)
        self.info = info

    def __str__(self):
        return self.info


class random_forest:
    def __init__(self):
        self.all_data = None
        self.train_data = None
        self.test_data = None
        self.train_acc = None
        self.test_acc = None
        self.trees = []

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
                self.all_data = np.concatenate((mfcc, zcr, rmse, np.array([[voice[1]]])), axis=1)
                fst = 1
            else:
                self.all_data = np.concatenate(
                    (self.all_data, np.concatenate((mfcc, zcr, rmse, np.array([[voice[1]]])), axis=1)),
                    axis=0)
        if self.all_data.shape == (9595, feature_num + 3):
            print("Data shape：")
            print(self.all_data.shape)
            self.all_data = self.all_data.tolist()
            # f = open("alldata.json", "w", encoding="utf-8")
            # json.dump(self.all_data, f)
        else:
            raise DataError("Data shape unmatched!")

    def get_all_data_np(self, path):
        f = open(path, "r", encoding="utf-8")
        self.all_data = json.load(f)
        if np.array(self.all_data).shape == (9595, 23):
            print("Data shape：")
            print(np.array(self.all_data).shape)
        else:
            raise DataError("Data shape unmatched!")

    def allocate_data(self, train_rate=0.88):
        l = len(self.all_data)
        self.train_data = self.all_data[:int(l * train_rate)]
        self.test_data = self.all_data[int(l * train_rate):]

    def get_invalid(self, feature_num):
        invalid = list(combinations([i for i in range(22)], feature_num))
        index = int(random.uniform(0, len(invalid)))
        return invalid[index]

    def bootstrap(self, invalid):
        add_set = set()
        test_set = set()
        train = []
        test = []
        sample_num = len(self.train_data)
        for i in range(sample_num):
            add_index = int(random.uniform(0, (sample_num - 1)))
            add_set.add(add_index)
            train.append(copy.deepcopy(self.train_data[add_index]))
        for i in range(sample_num):
            if i not in add_set:
                test_set.add(i)
                test.append(copy.deepcopy(self.train_data[i]))
        for k in train:
            for i in invalid:
                k[i] = 0
        return [train, test]

    def add_new_tree(self):
        invalid = self.get_invalid(10)
        train, test = self.bootstrap(invalid)
        my_tree = my_decision_tree.decision_tree()
        my_tree.train_data = train
        my_tree.test_data = test
        my_tree.get_tree_root()
        my_tree.acc()
        self.trees.append(my_tree)

    def generate_forest(self, tree_num):
        for i in range(tree_num):
            print("Tree " + str(i + 1) + " Operating")
            ant = timecnt(self.add_new_tree)
            ant()
            print("Done")

    def forest_predict(self, test_sample):
        vote = dict()
        for tree in self.trees:
            predict = tree.tree_predict(test_sample)
            if predict not in vote.keys():
                vote[predict] = 1
            else:
                vote[predict] += 1
        maxn = 0
        ans = -1
        for label in vote.keys():
            if vote[label] > maxn:
                maxn = vote[label]
                ans = label
        return ans

    def acc(self):
        print("Forest:")
        tot_train = len(self.train_data)
        tot_test = len(self.test_data)
        right_train = 0
        right_test = 0
        for sample in self.train_data:
            res = self.forest_predict(sample)
            if res == sample[-1]:
                right_train += 1
        for sample in self.test_data:
            res = self.forest_predict(sample)
            if res == sample[-1]:
                right_test += 1
        self.train_acc = right_train / tot_train
        self.test_acc = right_test / tot_test
        print("train acc", self.train_acc)
        print("test acc", self.test_acc)


def timecnt(func):
    def wrapper():
        st = time.time()
        print("Start: " + str(st))
        func()
        ed = time.time()
        print("End: " + str(ed))
        print("Total running time: " + str(ed - st))

    return wrapper


def main():
    forest = random_forest()
    path = r"alldata.json"
    # path = r"E:\SortedData"
    forest.get_all_data_np(path)
    # forest.get_all_data_p(path, 13)
    forest.allocate_data()
    forest.generate_forest(25)
    forest.acc()


if __name__ == '__main__':
    main()
