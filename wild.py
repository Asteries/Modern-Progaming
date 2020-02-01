import sys


# 函数读入n，返回一个二维列表v，v中包含所有合法的状态（包含了不可能的状态）
# v即是状态图中的结点
# 不判定不可能的情况，点孤立就是不可能
# v的每一个元素对应一个状态，每一个元素中的三个元素意义如下：
# v[i][0]状态i左岸修道士数量 v[i][1]状态i左岸野人数量 v[i][2] 状态i船的位置 1：船在左岸 0：船在右岸
def get_v(n):
    v = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(2):
                if i == 0 or i == n or i == j:
                    v.append([i, j, k])
    return v


# 函数读入此前得到的合法状态集合v和船一次可以运输的人数c，返回一个三维列表e，意义如下：
# i和j是状态在列表v中的索引
# e[i][j][0]:两种状态之间可不可以通过一次运输完成转化 0表示不可以 1表示可以
# e[i][j][1]:如果可以，运几个修道士
# e[i][j][2]:如果可以，运几个野人
# e即是状态图中的边
def get_e(v, c):
    l = len(v)
    e = [[[0, 0, 0] for i in range(l)] for j in range(l)]
    # 初始化e
    for i in range(l):
        for j in range(i + 1, l):
            # 遍历不同状态的两两组合，注意这里有对称的情况
            xd = v[i][0] - v[j][0]
            yr = v[i][1] - v[j][1]
            # xd：如果两状态可以转化，需要运输的修道士个数 yr：同理，野人个数
            if abs(xd + yr) <= c and (abs(xd) >= abs(yr) or xd == 0) and (xd != 0 or yr != 0):
                # 约束：船要载得下 修道士不能比野人少（除非是0个修道士） 没有人划船船不能开
                if (xd <= 0 and yr <= 0 and v[i][2] == 0 and v[j][2] == 1) or (
                        xd >= 0 and yr >= 0 and v[i][2] == 1 and v[j][2] == 0):
                    # 约束：在一次运输中，某一岸的修道士和野人同增同减
                    e[i][j][0] = 1
                    e[i][j][1] = abs(xd)
                    e[i][j][2] = abs(yr)
                    # 对称的运输也要记录，即能运过去肯定可以运回来
                    e[j][i][0] = 1
                    e[j][i][1] = abs(xd)
                    e[j][i][2] = abs(yr)
    return e


# 进行递归的深度优先搜索，给出状态[n,n,1]（v中的索引是len(v)-1）到状态[0,0,0]（v中的索引是0）的所有运输方法集合p（二维列表）
# p[i][j]表示第i+1种运输方法中的第j+1个状态
# 最初的起点bg，每一次递归搜索的起点st，终点（不会变）ed（都是v中的索引）
# v、e是此前得到的集合
# next是一维列表，next[i]:当前正在寻找的运输方法中，v中索引为i的状态的后继状态在v的索引
# vis是一维列表，vis[i]:v中索引为i的结点在当前搜索中有没有被访问过 0：没有 1：有
def dfs(bg, st, ed, v, e, p, next, vis):
    l = len(v)
    vis[st] = 1
    # 第一个结点标记为已经访问
    if st == ed:
        # 如果到达终点状态
        p.append([])
        # p中加入一个空列表表示一种运输方法
        k = bg
        # 从起点开始寻找后继
        while (k != ed):
            p[-1].append(k)
            # 把经过的状态不断加入刚刚加入p的那个列表里
            k = next[k]
        p[-1].append(k)
        return
        # 一个运输方法的搜索完成，返回继续
    for j in range(l):
        # 遍历所有结点
        if vis[j] == 0 and e[st][j][0] == 1:
            # 如果没有访问过并且当前起点状态st可以转化到j状态
            next[st] = j
            # st的后继标记为j
            dfs(bg, j, ed, v, e, p, next, vis)
            # 由于深度优先，递归，继续以j为st搜索
            vis[j] = 0
            # 等待搜索完全完成，应该访问其他st可以转化的状态了，回溯使得j变为没有访问过的情况


# 根据存储所有运输方法的集合p，找出最少运输次数的运输方法的集合minp
# minp是一维列表，minp[i]表示存储的i+1号最少运输次数方法在p中的索引
def get_shortest(p):
    if len(p) == 0:
        # 问题可能无解，如输入4 2
        print("没有可行的运输方法")
        sys.exit(0)
    minn = len(p[0])
    minp = [0]
    for r in range(1, len(p)):
        l = len(p[r])
        if l < minn:
            # 更短，清空minp
            minn = l
            minp.clear()
            minp.append(r)
        elif l == minn:
            # 相等，加入minp
            minp.append(r)
    return minp


# 打印所有运输方法和运输次数最少的方法
def print_path(minp, p, v):
    print("共有%d种运输方法：" % len(p))
    for x in p:
        for y in range(len(x)):
            print(v[x[y]], end="")
            if y != len(x) - 1:
                print("->", end="")
        print(" 运输次数：%d" % (len(x) - 1))
    print("运输最少的方法共有%d种:" % len(minp))
    for x in minp:
        for y in range(len(p[x])):
            print(v[p[x][y]], end="")
            if y != len(p[x]) - 1:
                print("->", end="")
        print(" 运输次数：%d" % (len(p[x]) - 1))


# 主函数
# p，next，vis列表需要在这里初始化
def main():
    n, c = [int(x) for x in input().split()]
    p = []
    v = get_v(n)
    l = len(v)
    next = [0] * l
    vis = [0] * l
    e = get_e(v, c)
    dfs(len(v) - 1, len(v) - 1, 0, v, e, p, next, vis)
    minp = get_shortest(p)
    print_path(minp, p, v)


if __name__ == '__main__': main()
