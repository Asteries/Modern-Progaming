import json


def print_stat(sum_sheet):
    with open('sums.json', 'r') as f:
        tot = json.load(f)
    maxc = []
    maxn = []
    minc = []
    minn = []
    avg = [0] * 30
    prov = sum_sheet[0].col_values(0, start_rowx=1, end_rowx=31)
    for i in range(19):
        max_temp = tot[i][0]
        max_ind = [0]
        min_temp = tot[i][0]
        min_ind = [0]
        for j in range(1, 30):
            if tot[i][j] > max_temp:
                max_temp = tot[i][j]
                max_ind.clear()
                max_ind.append(j)
            elif tot[i][j] == max_temp:
                max_ind.append(j)
            if tot[i][j] < min_temp:
                min_temp = tot[i][j]
                min_ind.clear()
                min_ind.append(j)
            elif tot[i][j] == min_temp:
                min_ind.append(j)
        maxc.append(max_ind)
        maxn.append(max_temp)
        minc.append(min_ind)
        minn.append(min_temp)

    for i in range(19):
        for j in range(30):
            avg[j] += tot[i][j]
    for i in range(30):
        avg[i] /= 19
    maxa = avg[0]
    mina = avg[0]
    maxi = [0]
    mini = [0]
    for i in range(1, 30):
        if avg[i] > maxa:
            maxa = avg[i]
            maxi.clear()
            maxi.append(i)
        elif avg[i] == maxa:
            maxi.append(i)
        if avg[i] < mina:
            mina = avg[i]
            mini.clear()
            mini.append(i)
        elif avg[i] == maxa:
            mini.append(i)

    for i in range(19):
        print(str(1997 + i) + "年排放CO2量最高的省份：")
        for c in maxc[i]:
            print(prov[c], end=" ")
        print()
        print("排放量为：" + str(maxn[i]))
        print(str(1997 + i) + "年排放CO2量最低的省份：")
        for c in minc[i]:
            print(prov[c], end=" ")
        print()
        print("排放量为：" + str(minn[i]))

    print("年平均CO2排放量最高的省份：")
    for c in maxi:
        print(prov[c], end=" ")
    print()
    print("排放量为：" + str(maxa))
    print("年平均CO2排放量最低的省份：")
    for c in mini:
        print(prov[c], end=" ")
    print()
    print("排放量为：" + str(mina))
