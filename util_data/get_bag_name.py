# coding=utf-8
import os
def funi_nofuni(dir):
    """
    函数功能:获取patch包名，目的在于看这些patch用到了哪些舌像图片
    :param dir: 文件目录
    :return: None
    """
    # 保存包名
    img_bags = []
    # 打开文件
    for img_name in os.listdir(dir):
        img_label,img_id,jpg= img_name.split(".") #按照'_'切割字符串
        img_bag,_ = img_id.split("_")
        img_bag = img_label+'.'+img_bag
        img_bags.append(img_bag)

    img_bags = list(set(img_bags))
    img_bags.sort()
    print(len(img_bags))
    file = open('bag_name_test.txt', 'w')
    file.write(str(img_bags))
    file.close()

if __name__ == "__main__":
    funi_nofuni("E:/project/rename/funi_nofuni/test")

