# coding=utf-8
import os
def funi_nofuni(dir):
    """
    函数功能:funi.B331.jpg -> funi.B033_1.jpg
    :param dir: 文件目录
    :return: None
    """
    # 打开文件
    for img_name in os.listdir(dir):
        label_name, img_id, jpg = img_name.split(".")                         #按照'.'切割字符串

        # # 2015数据集
        # bag_name, id = [img_id[i:i + 3] for i in range(0, len(img_id), 3)]  #获取图片的包名和id
        #
        # #包名字母后面加上0
        # bag_name = list(bag_name)     # str -> list
        # bag_name.insert(1, '0')
        # bag_name = ''.join(bag_name)  # list -> str

        # 2016数据集
        bag_name, id = [img_id[i:i + 4] for i in range(0, len(img_id), 4)]  # 获取图片的包名和id

        # 生成新的文件名
        new_img_name = label_name + '.' + bag_name + '_' + id + '.' + jpg
        print(img_name,new_img_name)
        Olddir = os.path.join(dir, img_name)  # 原来文件夹的路径
        Newdir = os.path.join(dir, new_img_name) #新的文件夹路径
        # print(Olddir,Newdir)
        # label_name,img_id,_= new_img_name.split(".")
        # img_bag,img_id = img_id.split('_')
        # img_id = img_bag + '_' + img_id
        #print(label_name,img_id, img_bag)
        os.rename(Olddir, Newdir)


if __name__ == "__main__":
    funi_nofuni("E:/project/rename/funi_nofuni/data/a")

