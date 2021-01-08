# coding=utf-8
import os
import random
import numpy as np

def get_img_dataset(funi_dir, no_funi_dir):
    """
    函数功能:获取原始大图图片名
    :param dir: 文件目录
    :return: None
    """
    # 保存腐腻图片名
    funi_names = []
    # 打开文件
    for img_name in os.listdir(funi_dir):
        img_label, funi_name, jpg = img_name.split(".")  # 按照'.'切割字符串
        funi_names.append(funi_name)
    list(set(funi_names))  # 去重
    random.shuffle(funi_names)
    print('腐腻图个数:%d' % len(funi_names))
    file = open('funi_names.txt', 'w')
    file.write(str(funi_names))
    file.close()

    # 保存非腐腻图片名
    no_funi_names = []
    # 打开文件
    for img_name in os.listdir(no_funi_dir):
        img_label, no_funi_name, jpg = img_name.split(".")  # 按照'.'切割字符串
        no_funi_names.append(no_funi_name)
    list(set(no_funi_names))  # 去重
    random.shuffle(no_funi_names)
    print('非腐腻图个数:%d' % len(no_funi_names))
    file = open('no_funi_names.txt', 'w')
    file.write(str(no_funi_names))
    file.close()

    return funi_names, no_funi_names


def split(funi_dataset, no_funi_dataset):
    '''
    将腐腻数据和非腐腻数据分别平均分成5份
    :param funi_dataset: 腐腻数据
    :param no_funi_dataset: 非腐腻数据
    :return: 切分后的腐腻数据和非腐腻数据
    '''
    split_size = 5

    # 腐腻图切分成5份
    # 每一份的大小
    each_funi_size = int(len(funi_dataset) / split_size)
    # 平均分成5份(每份17个，还剩三个，分别加在第0，1，2上面)
    split_funi_dataset = [funi_dataset[i:i + each_funi_size] for i in range(0, len(funi_dataset), each_funi_size)]
    for i in range(3):
        split_funi_dataset[i].append(split_funi_dataset[5][i])
    # 删除最后一个元素
    split_funi_dataset.pop()
    print('腐腻切分：%d,%d,%d,%d,%d' % (
    len(split_funi_dataset[0]), len(split_funi_dataset[1]), len(split_funi_dataset[2]), len(split_funi_dataset[3]),
    len(split_funi_dataset[4])))

    # 非腐腻图切成5份
    # 每一份的大小
    each_no_funi_size = int(len(no_funi_dataset) / split_size)
    # 平均分成5份(每份37个，还剩1个，加在第4上面)
    split_no_funi_dataset = [no_funi_dataset[i:i + each_no_funi_size] for i in
                             range(0, len(no_funi_dataset), each_no_funi_size)]
    split_no_funi_dataset[4].append(split_no_funi_dataset[5][0])
    # for i in range(3):
    #     split_no_funi_dataset[i+2].append(split_no_funi_dataset[5][i])
    # 删除最后一个元素
    split_no_funi_dataset.pop()
    print('非腐腻切分：%d,%d,%d,%d,%d' % (
        len(split_no_funi_dataset[0]), len(split_no_funi_dataset[1]), len(split_no_funi_dataset[2]),
        len(split_no_funi_dataset[3]),
        len(split_no_funi_dataset[4])))

    return split_funi_dataset, split_no_funi_dataset

def linkDataset(split_funi_dataset,split_no_funi_dataset):
    '''
    将切分后的腐腻数据和非腐腻数据整合，对应位置相加
    :param split_funi_dataset:
    :param split_no_funi_dataset:
    :return: 整合后的腐腻数据
    '''
    #用于存储整个数据
    datasets = []
    for i in range(5):
        dataset = split_funi_dataset[i] + split_no_funi_dataset[i]
        datasets.append(dataset)

    print('腐腻和非腐腻对应位置相加后：%d,%d,%d,%d,%d' %(len(datasets[0]),len(datasets[1]),len(datasets[2]),len(datasets[3]),len(datasets[4])))
    return datasets

def generateDataset(datasets):
    '''
    四份整合成训练集，剩余一份作为测试集
    :param datasets: 均分成5份的数据集
    :return:整合后的数据
    '''

    # 训练集
    train_dataset = []
    # 测试集
    test_dataset = []
    for i in range(5):
        train_data = []
        for j in range(5):
            if j != i:
                data1 = datasets[j]
                train_data.extend(data1)
        train_dataset.append(train_data)
        data2 = datasets[i]
        test_dataset.append(data2)
    print('整合后训练集：%d,%d,%d,%d,%d' %(len(train_dataset[0]),len(train_dataset[1]),len(train_dataset[2]),len(train_dataset[3]),len(train_dataset[4])))
    print('整合后测试集：%d,%d,%d,%d,%d' %(len(test_dataset[0]),len(test_dataset[1]),len(test_dataset[2]),len(test_dataset[3]),len(test_dataset[4])))

    # 整合后的数据集
    dataset = []
    for i in range(5):
        data2 = []
        data1 = train_dataset[i]
        data2.append(data1)
        data2.append(test_dataset[i])

        dataset.append(data2)
    # split0: 训练集和测试集
    print(len(dataset[0][0]), len(dataset[0][1]))
    return dataset

def split_feature(train_txt,test_txt,save_txt_path):
    '''
    将特征数据分成训练集和测试集
    :param train_txt: 手工切分数据做训练
    :param test_txt: 随机切分数据做测试
    :param save_txt_path: 保存路径
    :return:
    '''
    train_dataset = []
    test_dataset = []

    for i in range(5):
        train_data = []
        test_data = []
        #
        with open(train_txt, 'r') as fread1:
            for oneline in fread1:
                img_name, _ = oneline.strip().split("_")   # 按“_”切分数据，等到该数据属于的包名
                if img_name in dataset[i][0]:              # 包名在训练集中存在，将该条数据保存到训练集列表
                    train_data.append(oneline.strip())
            train_dataset.append(train_data)
        #
        with open(test_txt, 'r') as fread2:
            for oneline in fread2:
                img_name, _ = oneline.strip().split("_")  # 按“_”切分数据，等到该数据属于的包名
                if img_name in dataset[i][1]:  # 包名在测试集中存在，将该条数据保存到测试集列表
                    test_data.append(oneline.strip())
            test_dataset.append(test_data)



    train_all = np.array(train_dataset)
    test_all = np.array(test_dataset)
    for i in range(5):
        print(len(train_all[i]), len(test_all[i]))
    for j in range(5):
        np.savetxt(save_txt_path + "/train_" + str(j + 1) + '.txt', train_all[j], fmt="%s")
        np.savetxt(save_txt_path + "/test_" + str(j + 1) + '.txt', test_all[j], fmt="%s")


if __name__ == "__main__":
    # 获取原始大图图片名
    funi_dataset, no_funi_dataset = get_img_dataset("E:/project/rename/funi_nofuni/data_original/funi",
                                                    "E:/project/rename/funi_nofuni/data_original/no_funi")
    # 将腐腻数据和非腐腻数据分别平均分成5份
    split_funi_dataset,split_no_funi_dataset = split(funi_dataset, no_funi_dataset)
    # 将切分后的腐腻数据和非腐腻数据整合，对应位置相加
    datasets = linkDataset(split_funi_dataset, split_no_funi_dataset)
    # 四份整合成训练集，剩余一份作为测试集
    dataset = generateDataset(datasets)
    split_feature("split_data/vgg16_train_feature_sort.txt", "split_data/vgg16_test_feature_sort.txt", "split_data")



