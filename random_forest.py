import random
from itertools import combinations
import pandas as pd
import decision_tree
import json

# 构造森林中一棵树训练时使用的随机选择的数据
def randomlize_data(trainingData, list_to_delete):
    # 随机删去三分之一的样本数据
    sample_num = len(trainingData)
    delete = int(sample_num / 3)
    for i in range(0, delete):
        delete_index = int(random.uniform(0, (len(trainingData) - 1)))
        trainingData.pop(delete_index)
    # 根据传入的list选择这棵树可以使用的特征有哪些
    for k in trainingData:
        for i in list_to_delete:
            k[i] = 0  # 用来构造随机森林

    return trainingData

# 随机生成一个具有feature_num个特征的组合
def generate_random_feature_list(feature_num):
    zuhe = list(combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], feature_num))
    index = int(random.uniform(0,len(zuhe)))
    return zuhe[index]

# 根据森林预测结果
def predict_label_from_forest(forest,features):
    answear_list = dict()
    # 每棵树开始投票
    for tree in forest:
        predict = decision_tree.search_for_predict_label(features, tree)
        if predict not in answear_list.keys():
            answear_list[predict] = 1
        else:
            answear_list[predict] = answear_list[predict] + 1
    # 统计票数最多的结果
    most_voted = 0
    return_label = ''
    for label in answear_list.keys():
        if answear_list[label] > most_voted:
            most_voted = answear_list[label]
            return_label = label

    return return_label

if __name__ == '__main__':
    # 训练25棵树
    forest = list()
    for i in range(1,36):
        f = open("alldata.json", "r", encoding="utf-8")
        trainingData = json.load(f) # demo data from matlab
        featurelist = generate_random_feature_list(8)
        trainingData = randomlize_data(trainingData, featurelist)
        tree = decision_tree.build_decisiontree(trainingData)
        forest.append(tree)
        print(i)

    # 在训练集上测试正确率
    f = open("alldata.json", "r", encoding="utf-8")
    trainingData = json.load(f)  # demo data from matlab
    count = 0
    errorcount = 0
    sample_num = len(trainingData)
    for sample in trainingData:
        real_result = sample[-1]
        features = sample
        predict = predict_label_from_forest(forest,features)
        count = count + 1
        if predict != real_result:
            errorcount = errorcount + 1
    error_rate = float(errorcount / count)
    print(error_rate)

    # 对测试集做出预测并输出到csv
    dcHeadings, trainingData = decision_tree.loadCSV('test_for_other.csv')
    predict_label = list()
    for sample in trainingData:
        predict = predict_label_from_forest(forest,features)
        predict_label.append(predict)
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'income': predict_label})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(r"predict_labels_forest.csv", sep=',')
    print ("完成预测，并将结果存储到了 predict_labels_forest.csv")