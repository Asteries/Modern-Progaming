import sys
import jieba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def load_dict(text):
    jieba.load_userdict(text)


def get_emo_cnt(f):
    angry = [line.strip() for line in open(sys.argv[2], "r", encoding="UTF-8").readlines()]
    disgusted = [line.strip() for line in open(sys.argv[3], "r", encoding="UTF-8").readlines()]
    happy = [line.strip() for line in open(sys.argv[4], "r", encoding="UTF-8").readlines()]
    sad = [line.strip() for line in open(sys.argv[5], "r", encoding="UTF-8").readlines()]
    scared = [line.strip() for line in open(sys.argv[6], "r", encoding="UTF-8").readlines()]
    ecnt = [[0] * 5 for i in range(5000)]
    cnt = 0
    for l in f.readlines():
        lct = jieba.lcut(l)
        for w in lct:
            if w in angry:
                ecnt[cnt][0] += 1
            elif w in disgusted:
                ecnt[cnt][1] += 1
            elif w in happy:
                ecnt[cnt][2] += 1
            elif w in sad:
                ecnt[cnt][3] += 1
            elif w in scared:
                ecnt[cnt][4] += 1
        cnt += 1
    return ecnt


def get_emo(ecnt):
    emo = []
    for i in range(5000):
        maxn = 0
        maxi = []
        for j in range(5):
            if ecnt[i][j] > maxn:
                maxn = ecnt[i][j]
                maxi.clear()
                maxi.append(j)
            elif ecnt[i][j] == maxn:
                maxi.append(j)
        if len(maxi) == 5 and maxn == 0:
            emo.append([-1])
        else:
            emo.append(maxi)
    return emo


def convert_time(time):
    h = int(time[0:2])
    m = int(time[3:5])
    s = int(time[7:9])
    t = 60 * h + m + s / 60
    return t


def get_dim(f):
    dim = []
    for l in f.readlines():
        lfix = l.strip().split()
        co = convert_mars(float(lfix[-8]), float(lfix[-7]))
        dim.append([co[0], co[1], convert_time(lfix[-3])])
    return dim


def draw_3d(emo, dim):  #########
    colorlist = ["red", "purple", "orange", "gray", "black"]
    shapelist = ["o", "v", "s", "p", "h"]
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(5000):
        if emo[i][0] != -1:
            for j in range(len(emo[i])):
                ax.scatter(dim[i][0], dim[i][1], dim[i][2], s=5, marker=shapelist[emo[i][j]], c=colorlist[emo[i][j]])
    plt.savefig("em.png")


def convert_mars(lon_m, lat_m):
    a = 6378245.0
    e = 0.00669342162296594323
    lon = lon_m - 105
    lat = lat_m - 35
    delta_lon = 300 + lon + 2 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(math.fabs(lon)) + \
                (40 * math.sin(6 * lon * math.pi) + 40 * math.sin(2 * lon * math.pi)) / 3 + \
                (40 * math.sin(lon * math.pi) + 80 * math.sin(lon / 3 * math.pi)) / 3 + \
                (300 * math.sin(lon / 12 * math.pi) + 600 * math.sin(lon / 30 * math.pi)) / 3
    delta_lat = -100 + 0.2 * lon + 3 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(math.fabs(lon)) + \
                (40 * math.sin(6 * lon * math.pi) + 40 * math.sin(2 * lon * math.pi)) / 3 + \
                (40 * math.sin(lat * math.pi) + 80 * math.sin(lat / 3 * math.pi)) / 3 + \
                (320 * math.sin(lat / 12 * math.pi) + 640 * math.sin(lat / 30 * math.pi)) / 3
    m = 1 - e * math.sin(lat / 180 * math.pi) * math.sin(lat / 180 * math.pi)
    tlon = lon - (delta_lon * 180) / (a / (m ** 0.5) * math.cos(lat / 180 * math.pi) * math.pi)
    tlat = lat - (delta_lat * 180) / ((a * (1 - e)) / (m ** 1.5) * math.pi)
    return [tlon, tlat]


def main():
    f = open(sys.argv[1], "r", encoding="UTF-8")
    for i in range(2, 7):
        load_dict(sys.argv[i])
    ecnt = get_emo_cnt(f)
    print(ecnt[0])
    f.seek(0)
    emo = get_emo(ecnt)
    print(emo[0])
    dim = get_dim(f)
    print(dim[0])
    draw_3d(emo, dim)
    f.close()


if __name__ == '__main__': main()
