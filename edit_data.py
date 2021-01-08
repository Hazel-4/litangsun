# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

def get_txt_infos(info_txt):
    '''提取2016-2-7.txt中的信息
    :param info_txt: 文件信息存储的txt路径
    :return: 图片id,path,label
    '''
    # if mode not in ["train","test"]:
    # tang修改的代码(下替换上)
    img_ids = []
    img_labels = []
    img_paths = []
    with open(info_txt, "r", encoding="utf-8") as f_read:
        line = f_read.readline()
        while line:
            img_id, img_path, img_label = line.strip().split(",")
            img_ids.append(img_id)
            img_paths.append(img_path)
            img_labels.append(img_label)
            line = f_read.readline()
        f_read.close()
    return img_ids, img_paths, img_labels

def split_dataset(img_ids,img_paths,img_labels,save_path,split_size=5):
    '''一共有2391个样本,其中腐腻826个，非腐腻1565个：
        平均分成5份，每份478个样本:其中腐腻165个，非腐腻313个
    :param img_ids: 图片的id列表
    :param img_paths: 图片的路径列表
    :param img_labels: 图片的标签列表
    :param split_size: 划分个数：默认将数据集划分成5份
    :return: 划分后的所有数据
    '''

    data = pd.DataFrame({"img_id": img_ids, "img_path": img_paths, "img_label": img_labels},columns=['img_id', 'img_path', 'img_label'])

    # 对label进行分组
    group_data = data.groupby(by="img_label")
    # print(group_data.groups)

    for label_num, data in group_data:
        # if label_num == 0:
        if label_num == "funi":
            funi_dataset = data

        # if label_num == 1:
        if label_num == "no_funi":
            no_funi_dataset = data
    # 打乱顺序
    funi_dataset = shuffle(funi_dataset)
    no_funi_dataset = shuffle(no_funi_dataset)

    # 将数据中腐腻和非腐腻均等分成5份
    # each_size = int(len(img_ids) / split_size)  # 每一份数据的大小
    each_funi_size = int(len(funi_dataset) / split_size)    # 每一份数据中腐腻的大小
    each_no_funi_size = int(len(no_funi_dataset) / split_size)  # 每一份数据中非腐腻的大小
    # 用来保存所有的数据
    split_all = []

    for i in range(split_size):
        each_split = shuffle(pd.concat([funi_dataset.iloc[each_funi_size * i:each_funi_size * (i+1), :], no_funi_dataset.iloc[each_no_funi_size * i:each_no_funi_size * (i+1), :]], axis=0))
        # each_split.to_csv("data/cross_validation_txt/split5_1/split_" + str(i+1) + '.csv')
        np.savetxt(save_path + "/split_" + str(i+1) + '.txt', each_split, fmt="%s", delimiter='\t')  # 输出每一份数据
        split_all.append(each_split)
    return split_all


def generateDataset(datadir, outdir):
    '''从切分的数据集中，对其中四份抽样汇成一个;
    剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    :param datadir: 切分后的文件输出目录:split5_1
    :param outdir:  抽样的数据集存放目录:cross_validation_dataset
    :return:所有的训练集和测试集
    '''

    if not os.path.exists(outdir):  # if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    # 将字符串标签转化成数字
    label_name_to_num = {"no_funi": 0, "funi": 1}
    train_all = []
    test_all = []
    cross_now = 0
    for eachfile1 in listfile:
        train_ids = []
        train_paths = []
        train_labels = []

        test_ids = []
        test_paths = []
        test_labels = []

        cross_now += 1
        # 保存训练集
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:
                with open(datadir + '/' + eachfile2, 'r') as fr_trainsets:
                    for oneline_train in fr_trainsets:
                        id, path, label = oneline_train.strip().split("\t")
                        train_ids.append(id)
                        train_paths.append(path)
                        train_labels.append(label_name_to_num[label])

        train_dataset = pd.DataFrame({"img_id": train_ids, "img_path": train_paths, "img_label": train_labels})
        train_dataset.to_csv(outdir + '/train_' + str(cross_now) + ".csv")
        train_all.append(train_dataset)

        # 保存测试集
        with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
            for oneline_test in fr_testsets:
                id, path, label = oneline_test.strip().split("\t")
                test_ids.append(id)
                test_paths.append(path)
                test_labels.append(label_name_to_num[label])
        test_dataset = pd.DataFrame({"img_id": test_ids, "img_path": test_paths, "img_label": test_labels})
        test_dataset.to_csv(outdir + '/test_' + str(cross_now) + ".csv")
        test_all.append(test_dataset)

    return train_all, test_all


def NN_test_bag(data):
    '''
    :param data: pd.DataFrame({"id": test_img_ids, "pre_label": test_pred_labels,"true_label": true_label})
    :return: ACC TPR TNR
    '''
    test_ids = data.test_img_ids
    print(type(test_ids))


if __name__ == "__main__":

    # 将数据等分成5份
    # save_path = "data/txt/2016-2-7_txt/split5_1"
    # img_ids, img_paths, img_labels = get_txt_infos("data/txt/2016-2-7_txt/2016-2-7.txt")
    # split_all = split_dataset(img_ids, img_paths, img_labels, save_path)

    #将数据分成测试集和验证集
    datadir = "data/txt/2016-2-7_txt/split5_1"
    outdir = "data/txt/2016-2-7_txt/train_test"
    train_all, test_all = generateDataset(datadir, outdir)

