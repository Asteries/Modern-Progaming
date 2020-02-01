import csv
import pandas as pd

class Tree:
    def __init__(self, divide_value=None, trueBranch=None, falseBranch=None, results=None, col=-1, data=None):
        self.divide_value = divide_value    #产生分叉的特征值
        self.trueBranch = trueBranch        #对产生分叉的特征值，取值为真的子树
        self.falseBranch = falseBranch      #对产生分叉的特征值，取值为假的子树
        self.results = results          #如果是叶子节点，则记录这个节点上不同label样本的个数，否则为None
        self.col_index = col            #产生分叉的特征值的索引
        self.data = data                #这一个节点拥有的所有样本的数据

def get_num_of_each_label_type(data):
    # 输出data里每种label的数目
    label_count_for_each_type = dict()

    for feature_and_label in data:
        label = feature_and_label[-1]
        if label not in label_count_for_each_type:
            label_count_for_each_type[label] = 1
        else:
            label_count_for_each_type[label] += 1

    return label_count_for_each_type

def Gini(data):
    #Gini（D） = 1 - 求和（pk^2）
    total_sample_num = len(data)
    probility_list = get_num_of_each_label_type(data)
    probility_squre = 0.0
    for i in probility_list:
        probility_squre += (probility_list[i] / total_sample_num) * (probility_list[i] / total_sample_num)
    gini = 1 - probility_squre
    return gini

def split_data_by_divide_value(data, divide_value, column):
    # 根据divide_value，把data分成两份
    true_list = list()
    false_list = list()
    if isinstance(divide_value, str):  # for String type
        for sample in data:
            if sample[column] == divide_value:
                true_list.append(sample)
            else:
                false_list.append(sample)
    elif (isinstance(divide_value, int) or isinstance(divide_value, float)):  # for int and float type
        for sample in data:
            if (sample[column] >= divide_value):
                true_list.append(sample)
            else:
                false_list.append(sample)

    return (true_list, false_list)


def build_decisiontree(data):
    Gain_D = Gini(data)
    feature_num = len(data[0])
    sample_num = len(data)
    best_gain = 0.0
    best_value = None
    best_set = None

    # 循环每一种特征，找到最合适的
    for col in range(1,feature_num - 1):
        # 忽略编号特征,只针对这一个题目有用
        if col == 3:
            continue
        #找到某一个特征的取值组合
        feature_value_set = set([feature_col[col] for feature_col in data])
        # 找到某一个特征的取值组合中的每一个取值作为划分点的gini
        for divide_value in feature_value_set:
            truelist, falselist = split_data_by_divide_value(data, divide_value, col)
            ratio_true_set = len(truelist) / sample_num
            # gain大说明划分后的gini指数反而小
            gain = Gain_D - ratio_true_set * Gini(truelist) - (1 - ratio_true_set) * Gini(falselist)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, divide_value)
                best_set = (truelist, falselist)

    #成为叶子节点还是继续分叉
    if best_gain <= 0:
        return Tree(results=get_num_of_each_label_type(data),  data=data)
    elif best_gain > 0:
        trueBranch = build_decisiontree(best_set[0])
        falseBranch = build_decisiontree(best_set[1])
        return Tree(col=best_value[0], divide_value=best_value[1], trueBranch=trueBranch, falseBranch=falseBranch)

def search_for_predict_label(data, tree):
    # 下面还有树
    if tree.results == None:
        next_node = None
        v = data[tree.col_index]
        if isinstance(v, str):
            if v == tree.divide_value:
                next_node = tree.trueBranch
            else:
                next_node = tree.falseBranch
        elif isinstance(v, int) or isinstance(v, float):
            if v >= tree.divide_value:
                next_node = tree.trueBranch
            else:
                next_node = tree.falseBranch
        return search_for_predict_label(data, next_node)
    # 是叶子节点
    elif tree.results != None:
        most = 0
        key = None
        for k in tree.results.keys():
            if tree.results[k] > most:
                most = tree.results[k]
                key = k
        return key

def loadCSV(file):
    def tran2number_if_need(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt'))
    head = {}

    heads = next(reader)
    for index, head_name in enumerate(heads):
            index_name = 'Column %d' % index
            head[index_name] = str(head_name)

    data = [[tran2number_if_need(item) for item in row] for row in reader]

    return head, data

if __name__ == '__main__':
    # the bigger example
    dcHeadings, trainingData = loadCSV('train_for_other.csv')
    decisionTree = build_decisiontree(trainingData)
    print("决策树建立成功")

    # 在原训练集上测试准确率
    dcHeadings, trainingData = loadCSV('train_for_other.csv') # demo data from matlab
    # trainingData[1][-1] = trainingData[1][-1]*-1
    count = 0
    errorcount = 0
    sample_num = len(trainingData)
    for sample in trainingData:
        real_result = sample[-1]
        features = sample[0:-1]
        predict = search_for_predict_label(features, decisionTree)
        count = count + 1
        if predict != real_result:
            errorcount = errorcount + 1
    error_rate = float(errorcount/count)
    print(error_rate)

    # 对test进行预测，并将代码输出到csv
    dcHeadings, trainingData = loadCSV('test_for_other.csv')
    predict_label = list()
    for sample in trainingData:
        predict = search_for_predict_label(sample, decisionTree)
        predict_label.append(predict)
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'predict': predict_label})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(r"predict_labels.csv", sep=',')
    print ("完成预测，并将结果存储到了 predict_labels.csv")