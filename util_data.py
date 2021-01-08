# coding=utf-8
import pandas as pd
from sklearn.utils import shuffle


def get_img_infos(mode, img_info_txt, label_name_to_num=None):
    '''读取txt中存储的图片信息
    :param mode: train or test
    :param img_info_txt: 文件信息存储的txt路径
    :param label_name_to_num: 将字符串标签转为数字
    :return: 图片id信息和图片的标签(mode为train时不为空,mode为test时为空)
    '''
    # if mode not in ["train","test"]:
    # tang修改的代码(下替换上)
    if mode not in ["train", "test", "extract_features"]:
        raise ValueError("mode:%s,is train,test or extract_features" % mode)
    img_ids = []
    img_labels = []
    img_paths = []
    with open(img_info_txt, "r", encoding="utf-8") as f_read:
        line = f_read.readline()
        while line:
            if mode == "train":
                img_id, img_path, label = line.replace("\n", "").split(",")
                if label_name_to_num != None:
                    img_labels.append(label_name_to_num[label])
                else:
                    img_labels.append(label)
            elif mode == "test":
                img_id, img_path, label = line.replace("\n", "").split(",")

                # tang增加的代码{
            elif mode == "extract_features":
                img_id, img_path, label = line.replace("\n", "").split(",")
                if label_name_to_num != None:
                    img_labels.append(label_name_to_num[label])
                else:
                    img_labels.append(label)
            # tang增加的代码}

            # img_ids.append(int(img_id))
            img_ids.append(img_id)
            img_paths.append(img_path)
            line = f_read.readline()
        f_read.close()
    return img_ids, img_labels, img_paths


def split_dataset(img_ids, img_paths, img_labels, val_size=0.1):
    '''
    :param img_ids: 图片的id列表
    :param img_paths: 图片的路径列表	
    :param img_labels: 图片的标签列表
    :param val_size: 验证集大小5000
    :param test_size: 测试集大小10000
    :return: 训练集数据,验证集数据,测试集数据
    '''

    data = pd.DataFrame({"img_id": img_ids, "img_path": img_paths, "img_label": img_labels})

    # 对label进行分组
    group_data = data.groupby(by="img_label")
    print(group_data)

    for label_num, data in group_data:
        if label_num == 0:
        # if label_num == "funi":
            cat_dataset = data

        if label_num == 1:
        # if label_num == "no_funi":
            dog_dataset = data
    # 打乱顺序
    cat_dataset = shuffle(cat_dataset)
    dog_dataset = shuffle(dog_dataset)
    val_num = int(len(img_ids) * val_size)
    # print(val_num)
    val_dataset = shuffle(pd.concat([cat_dataset.iloc[:val_num, :], dog_dataset.iloc[:val_num, :]], axis=0))
    train_dataset = shuffle(pd.concat([cat_dataset.iloc[val_num:, :], dog_dataset.iloc[val_num:, :]], axis=0))
    # 将训练集文件和测试集文件保存为csv文件
    # train_dataset.to_csv("../data/txt/2016-2-7_txt/train.csv")
    # val_dataset.to_csv("../data/txt/2016-2-7_txt/val.csv")
    return train_dataset, val_dataset

def get_csv_info(info_csv):
    '''
    获取csv文件信息：交叉验证集信息
    :param info_csv: csv文件路径
    :return:
    '''
    csv_data = pd.read_csv(info_csv)
    return csv_data

# if __name__ == "__main__":
    # train_img_ids,train_img_labels,img_paths = get_img_infos("train","data/txt/train.txt")
    # csv_data = get_csv_info('data/cross_validation_txt/train_test/test_1.csv')



