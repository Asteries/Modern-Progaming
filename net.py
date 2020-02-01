import networkx as nx
import jieba
import matplotlib.pyplot as plt
from networkx.algorithms.community import k_clique_communities


class WordNetwork(nx.Graph):
    #初始化边和点
    def get_word_net(self, linecut, w):
        for x in linecut:
            self.add_nodes_from(x)
        for x in linecut:
            for i in range(len(x) - 1):
                for z in range(i + 1, min(i + w + 1, len(x))):
                    if (x[z], x[i]) not in self.edges:
                        self.add_edge(x[z], x[i], weight=1)
                    else:
                        self[x[z]][x[i]]['weight'] += 1
    #去除权重不足t的边
    def del_sm_edge(self, t):
        del_e = []
        for x in self.edges:
            if self[x[0]][x[1]]['weight'] < t:
                del_e.append((x[0], x[1]))
        for x in del_e:
            self.remove_edge(x[0], x[1])
    #去除度是0的结点
    def del_em_node(self):
        del_n = []
        for x in self.nodes:
            if self.degree[x] == 0:
                del_n.append(x)
        for x in del_n:
            self.remove_node(x)
    #获取topn
    def topn(self, n):
        self.stn = []
        pr = nx.pagerank(self)
        h_hub = nx.hits(self)[0]
        for x in self.nodes:
            self.stn.append((x, self.degree[x], pr[x], h_hub[x]))
        self.stn.sort(key=lambda s: (-s[1], -s[2], -s[3]))
        print(self.stn[0:n])

    #发现社区
    def community(self, k):
        self.k_co = list(k_clique_communities(self, k))
        print(self.k_co)

#分词
def word_cut(filepath, stop):
    f = open(filepath, "r", encoding='utf-8')
    line = f.readlines()
    c5 = []
    c1 = []
    for i in range(len(line)):
        if line[i][-2] == '1':
            l = jieba.lcut(line[i][0:-3])
            c5.append([])
            for x in l:
                if x not in stop and x != ' ':
                    c5[-1].append(x)
        else:
            l = jieba.lcut(line[i][0:-3])
            c1.append([])
            for x in l:
                if x not in stop and x != ' ':
                    c1[-1].append(x)
    return [c5, c1]


def get_stop(stopfile):
    stop = [line.strip() for line in open(stopfile, 'r', encoding='utf-8').readlines()]
    return set(stop)

#给出共有词
def common(net1, net2):
    c = []
    for x in net1.stn:
        for y in net2.stn:
            if x[0] == y[0]:
                c.append([x, y])
    return c

#建立网络
def build(g5, g1, c5, c1, w, t):
    g5.get_word_net(c5, w)
    g5.del_sm_edge(t)
    g5.del_em_node()
    g1.get_word_net(c1, w)
    g1.del_sm_edge(t)
    g1.del_em_node()
    print(len(g5.edges))
    print(len(g5.nodes))
    print(len(g1.edges))
    print(len(g1.nodes))

#比较top25
def compare51(g5, g1):
    print("5分的top25：")
    g5.topn(25)
    print("1分的top25：")
    g1.topn(25)
    c = common(g5, g1)
    print("共用词及其重要性的比较：")
    print(c)
    return c

#可视化共有词的连接结构
def visial_common(g5, g1, c):
    nc5 = [y[0][0] for y in c]
    nc1 = [y[0][0] for y in c]
    for x in [y[0][0] for y in c]:
        for z in g5.nodes:
            if (x, z) in g5.edges or (z, x) in g5.edges:
                nc5.append(z)
    for x in [y[0][0] for y in c]:
        for z in g1.nodes:
            if (x, z) in g1.edges or (z, x) in g1.edges:
                nc1.append(z)
    plt.subplot(1, 2, 1)
    nx.draw_networkx(g5, with_labels=True, pos=nx.shell_layout(g5), nodelist=nc5)
    #labels = nx.get_edge_attributes(g5, 'weight')
    #nx.draw_networkx_edge_labels(g5, pos=nx.shell_layout(g5), edge_labels=labels)
    plt.subplot(1, 2, 2)
    nx.draw_networkx(g1, with_labels=True, pos=nx.shell_layout(g1), nodelist=nc1)
    #labels = nx.get_edge_attributes(g1, 'weight')
    #nx.draw_networkx_edge_labels(g1, pos=nx.shell_layout(g1), edge_labels=labels)
    plt.savefig('netcomm.png')

#社区发现
def find_co(g5, g1):
    print("5分的社区发现：")
    g5.community(4)
    print("1分的社区发现：")
    g1.community(4)


def main():
    plt.figure(figsize=(10, 4.8))
    g5 = WordNetwork()
    g1 = WordNetwork()
    stop = get_stop("C:\\Users\\Asteries\\PycharmProjects\\OO\\stopwords_list.txt")
    c5, c1 = word_cut("C:\\Users\\Asteries\\PycharmProjects\\OO\\1分和5分评论文本.txt", stop)
    build(g5, g1, c5, c1, 7, 300)
    print("Built!")
    c = compare51(g5, g1)
    visial_common(g5, g1, c)
    find_co(g5, g1)
    nx.write_gexf(g5, 'g5.gexf')
    nx.write_gexf(g1, 'g1.gexf')


if __name__ == "__main__":
    main()
