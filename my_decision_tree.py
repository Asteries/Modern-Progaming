import librosa
import os
from random import shuffle
import numpy as np
import json
from tqdm import tqdm


class DataError(Exception):
    def __init__(self, info):
        super().__init__(self)
        self.info = info

    def __str__(self):
        return self.info


class decision_tree:
    def __init__(self):
        self.all_data = None
        self.train_data = None
        self.test_data = None
        self.tree_root = None
        self.train_acc = None
        self.test_acc = None

    def allocate_data(self, train_rate=0.88):
        l = len(self.all_data)
        self.train_data = self.all_data[:int(l * train_rate)]
        self.test_data = self.all_data[int(l * train_rate):]

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
            f = open("alldata.json", "w", encoding="utf-8")
            json.dump(self.all_data, f)
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

    def get_label_cnt(self, data):
        label_cnt = {}
        for sample in data:
            label = sample[-1]
            if label not in label_cnt:
                label_cnt[label] = 1
            else:
                label_cnt[label] += 1
        return label_cnt

    def gini(self, data):
        sample_num = len(data)
        prob_list = self.get_label_cnt(data)
        prob_square_sum = 0
        for i in prob_list:
            prob_square_sum += (prob_list[i] / sample_num) * (prob_list[i] / sample_num)
        return 1 - prob_square_sum

    def split_data(self, data, limit, attribute):
        true_list = []
        false_list = []
        for sample in data:
            if sample[attribute] >= limit:
                true_list.append(sample)
            else:
                false_list.append(sample)
        return [true_list, false_list]

    def build_decision_tree(self, data=None):
        if data == None:
            data = self.train_data
        gini_d = self.gini(data)
        feature_num = len(data[0]) - 1
        sample_num = len(data)
        best_gain = 0.0
        best_value = None
        best_set = None
        attcnt = 0
        for attribute in range(feature_num):
            value_set = set([sample[attribute] for sample in data])
            valuecnt = 0
            l = len(value_set)
            for limit in value_set:
                true_list, false_list = self.split_data(data, limit, attribute)
                true_rate = len(true_list) / sample_num
                gini_da_sum = true_rate * self.gini(true_list) + (1 - true_rate) * self.gini(false_list)
                gain = gini_d - gini_da_sum
                if gain > best_gain:
                    best_gain = gain
                    best_value = (attribute, limit)
                    best_set = (true_list, false_list)
                valuecnt += 1

                if valuecnt % 1000 == 0:
                   print(
                       "limit:" + str(valuecnt) + "/" + str(l) + " attribute:" + str(attcnt + 1) + "/" + str(
                           feature_num))
            attcnt += 1
        if best_gain <= 0:
            return tree_node(results=self.get_label_cnt(data), data=data)
        elif best_gain > 0:
            true_tree = self.build_decision_tree(data=best_set[0])
            false_tree = self.build_decision_tree(data=best_set[1])
            return tree_node(attribute=best_value[0], limit=best_value[1], true_tree=true_tree,
                             false_tree=false_tree)

    def get_tree_root(self):
        self.tree_root = self.build_decision_tree()

    def tree_predict(self, test_sample, tree=None):
        if tree == None:
            tree = self.tree_root
        if tree.results == None:
            value = test_sample[tree.attribute]
            if value >= tree.limit:
                next = tree.true_tree
            else:
                next = tree.false_tree
            return self.tree_predict(test_sample, next)
        elif tree.results != None:
            maxn = 0
            ans = None
            for key in tree.results.keys():
                if tree.results[key] > maxn:
                    maxn = tree.results[key]
                    ans = key
            return ans

    def acc(self):
        tot_train = len(self.train_data)
        tot_test = len(self.test_data)
        right_train = 0
        right_test = 0
        for sample in self.train_data:
            res = self.tree_predict(sample)
            if res == sample[-1]:
                right_train += 1
        for sample in self.test_data:
            res = self.tree_predict(sample)
            if res == sample[-1]:
                right_test += 1
        self.train_acc = right_train / tot_train
        self.test_acc = right_test / tot_test
        print("train acc", self.train_acc)
        print("test acc", self.test_acc)


class tree_node:
    def __init__(self, limit=None, true_tree=None, false_tree=None, results=None, attribute=None, data=None):
        self.limit = limit
        self.true_tree = true_tree
        self.false_tree = false_tree
        self.results = results
        self.attribute = attribute
        self.data = data


def main():
    path = r"alldata.json"
    # path = r"E:\SortedData"
    my_tree = decision_tree()
    my_tree.get_all_data_np(path)
    # my_tree.get_all_data_p(path, 20)
    print("Got")
    my_tree.allocate_data()
    print("Allocated")
    my_tree.get_tree_root()
    print("Built")
    my_tree.acc()


if __name__ == "__main__":
    main()
