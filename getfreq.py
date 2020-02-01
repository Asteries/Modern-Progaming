import sys
import os
import string
import collections

import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def load_stoppping_words(filepath):
    sws = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
    return set(sws)


def get_terms(filepath, stoppingwords):
    ws = []
    with open(filepath, 'r', encoding='UTF-8') as f:
        for line in f:
            for w in jieba.cut(line.strip()):
                if w not in stoppingwords and w != ' ':
                    ws.append(w)
    return ws


def get_terms_freq(ws):
    wf = {}
    for w in ws:
        if w in wf:
            wf[w] += 1
        else:
            wf[w] = 1
    return wf


def get_features(wf, high=100000, low=5):
    features = []
    for k in wf:
        if wf[k] <= high and wf[k] >= low:
            features.append(k)
    return features


def get_vs_model_weight(wf, features):
    v = []
    for f in features:
        if f in wf:
            v.append(wf[f])
        else:
            v.append(0)
    return v


def get_vs_model_onehot(wf, features):
    v = []
    for f in features:
        if f in wf:
            v.append(1)
        else:
            v.append(0)
    return v


def get_topk_terms(wf, topn=10):
    tws = []
    sorted_dic = collections.OrderedDict(sorted(wf.items(), \
                                                key=lambda d: d[1], reverse=True))
    for k in list(sorted_dic.keys())[0:topn]:
        tws.append(k)
    return tws


def draw_cloud(tws, wf):
    freq = {}
    for t in tws:
        freq[t] = wf[t]
    wordcloud = WordCloud(font_path='msyh.ttc', max_font_size=40, \
                          width=1000, height=700, background_color='white').generate_from_frequencies(freq)
    wordcloud.to_file('wc1.png')


def get_vectors(filepath, features, stoppingwords, v2):
    vectors = []
    l = len(features)
    s = 0
    for i in v2:
        s += i
    for i in range(len(v2)):
        v2[i] /= s
    with open(filepath, 'r', encoding='UTF-8') as f:
        for line in f:
            vectors.append([0] * l)
            linecut = jieba.lcut(line.strip())  # 使用lcut，直接给出list
            for i in range(l):
                for c in linecut:
                    if c not in stoppingwords and c == features[i] and c != ' ':
                        vectors[-1][i] += 1
            for i in range(l):
                vectors[-1][i] *= v2[i]  # 加权
    return vectors


def get_euclid(vectors):
    l = len(vectors)
    euclid = [[0] * l for x in range(l)]
    for i in range(l):
        j = i + 1
        while (j < l):
            for k in range(len(vectors[0])):
                euclid[i][j] += (vectors[i][k] - vectors[j][k]) ** 2
            euclid[i][j] = euclid[i][j] ** 0.5
            euclid[j][i] = euclid[i][j]
            j += 1
    return euclid


def get_center(vectors):
    l = len(vectors[0])
    center = [0] * l
    for i in range(l):
        for j in vectors:
            center[i] += j[i]
        center[i] /= l
    return center


def get_rep(vectors, center, filepath):
    minn = 999999999
    no = -1
    for i in range(len(vectors)):
        dis = 0
        for k in range(len(vectors[0])):
            dis += (vectors[i][k] - center[k]) ** 2
        dis = dis ** 0.5
        if dis < minn:
            minn = dis
            no = i
    with open(filepath, 'r', encoding='UTF-8') as f:
        i = 0
        for line in f:
            if i == no:
                print(line)
                break
            i += 1


def draw_bar(tws, wf):
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12.8, 19.2))
    freq = {}
    for t in tws:
        freq[t] = wf[t]
    b = plt.barh(range(len(freq)), list(freq.values()), height=0.5, tick_label=list(freq.keys()))
    plt.savefig("bc1.png")
    plt.close()


def main():
    if len(sys.argv) != 3:
        print("Usage: python xxx.py spfile docfile")
    else:
        stoppingwords = load_stoppping_words(sys.argv[1])  # 读入停词表
        # print(len(stoppingwords))
        ws = get_terms(sys.argv[2], stoppingwords)  # 切割句子
        # print(len(ws))
        wf = get_terms_freq(ws)  # 记录词频
        print(wf)
        print(len(wf))
        features = get_features(wf, high=100000, low=30)  # 给出特征集
        print(features)
        print(len(features))
        v1 = get_vs_model_onehot(wf, features)  # 一位有效编码
        # print(v1)
        v2 = get_vs_model_weight(wf, features)  # 各特征词的权重
        print(v2)
        tws = get_topk_terms(wf, 100)  # top100高频词
        print(tws)
        draw_cloud(tws, wf)  # 词云图
        draw_bar(tws, wf)  # 柱状图
        vectors = get_vectors(sys.argv[2], features, stoppingwords, v2)
        print(vectors[0])
        euclid = get_euclid(vectors)
        print(euclid[0])
        center = get_center(vectors)
        get_rep(vectors, center, sys.argv[2])  # 找与重心距离最小的评论作为代表性评论


if __name__ == '__main__': main()
