import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')

sys.path.append('../')
from DataGenerator import ImageDataGenerator
from util_data import get_img_infos,split_dataset
import split_dataset_2016
from datetime import datetime
import pandas as pd

# 设置训练文件路径
CNN_txt = "../data/2016_txt/CNN.txt"
MISVM_txt = "../data/2016_txt/MISVM.txt"
test_txt = "../data/2016_txt/MISVM.txt"

num_classes = 2
resnet_depth = 50
# train_layers = "fc,scale5/block3"
train_layers = "fc"
batch_size = 32
num_epochs = 30
learning_rate = 0.00001

# batch_size = 64
# num_epochs = 10
# learning_rate = 0.00001

multi_scale = ''
# 训练多少次保存tensor board
display_step = 20


tensorboard_dir = "../tensorboard/resnet"
tensorboard_train_dir = "../tensorboard/resnet/train"
tensorboard_val_dir = "../tensorboard/resnet/val"
checkpoint_dir = "../checkpoints/resnet"

# Placeholders
# x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
is_training = tf.placeholder('bool', [])

model2 = ResNetModel(is_training, depth=resnet_depth, num_classes=num_classes)
# output_y = model2.inference(x)

def train():
    label_name_to_num = {"no_funi": 0, "funi": 1}
    # 获取所有的训练数据
    img_ids, img_labels, img_paths = get_img_infos("train", CNN_txt, label_name_to_num)
    train_dataset, val_dataset = split_dataset(img_ids, img_paths, img_labels)
    with tf.device("/cpu:0"):
        train_data = ImageDataGenerator(train_dataset, mode="train", batch_size=batch_size, num_classes=num_classes,
                                        shuffle=True)
        val_data = ImageDataGenerator(val_dataset, mode="val", batch_size=batch_size, num_classes=num_classes)
        # 创建一个获取下一个batch的迭代器
        iterator = tf.data.Iterator.from_structure(train_data.data.output_types, train_data.data.output_shapes)
        next_batch = iterator.get_next()

    # 初始化训练集数据
    training_init_op = iterator.make_initializer(train_data.data)
    # 初始化测试集数据
    val_init_op = iterator.make_initializer(val_data.data)
    # 获取需要重新训练的变量
    var_list = [v for v in tf.trainable_variables() if v.name.split("/")[0] in train_layers]

    # train_layers = FLAGS.train_layers.split(',')

    loss = model2.loss(x, y)
    # # 定义交叉熵损失值
    # with tf.name_scope("cross_entropy_loss"):
    #     # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_y, labels=y))
    #     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_y, labels=y)
    #     cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #     regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #     loss = tf.add_n([cross_entropy_mean] + regularization_losses)
    # train_op = model2.optimize(learning_rate, train_layers)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model2.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver(max_to_keep=50)

    # Batch preprocessors
    # multi_scale = FLAGS.multi_scale.split(',')
    # if len(multi_scale) == 2:
    #     multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    # else:
    #     multi_scale = None

    # Get the number of training/validation steps per epoch
    # 计算每轮的迭代次数
    train_batches_per_epoch = int(np.floor(train_data.data_size / batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        model2.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), tensorboard_dir))

        for epoch in range(num_epochs):
            sess.run(training_init_op)
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Start training
            for step in range(train_batches_per_epoch):
                batch_xs, batch_ys = sess.run(next_batch)
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, is_training: True})

                # Logging
                if step % display_step == 0:
                    # s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                    s, train_acc, train_loss = sess.run([merged_summary, accuracy, loss], feed_dict={x: batch_xs, y: batch_ys, is_training: False})

                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)
            sess.run(val_init_op)
            # Epoch completed, start validation
            print("{} Start validation".format(datetime.now()))
            test_acc = 0.
            test_count = 0
            test_loss = 0
            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = sess.run(next_batch)
                # acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, is_training: False})
                acc, test_batch_loss = sess.run([accuracy, loss], feed_dict={x: batch_tx, y: batch_ty, is_training: False})
                test_acc += acc
                test_count += 1
                test_loss += test_batch_loss
            test_acc /= test_count
            test_loss /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
            ])
            val_writer.add_summary(s, epoch+1)
            print("{} train_acc = {:.4f}, train_loss = {:.4f}, test_acc = {:.4f}, test_loss = {:.4f}".format(datetime.now(), train_acc, train_loss, test_acc, test_loss))

            # Reset the dataset pointers
            # val_preprocessor.reset_pointer()
            # train_preprocessor.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.now()))

            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, "model_epoch%s_%.4f.ckpt" % (str(epoch + 1), test_acc))
            saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_path))




# 预测测试集生成结果
def genrate_pre_result():
    label_name_to_num = {"no_funi": 0, "funi": 1}
    # 获取需要预测结果的所有数据
    img_ids, true_label, img_paths = get_img_infos("train", test_txt, label_name_to_num)
    test_dataset = pd.DataFrame({"img_id": img_ids, "img_path": img_paths})
    with tf.device("/cpu:0"):
        test_data = ImageDataGenerator(test_dataset, mode="test", batch_size=batch_size, num_classes=num_classes)
        iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        next_batch = iterator.get_next()
    # 初始化测试集中的图片数据
    test_init_op = iterator.make_initializer(test_data.data)

    output_y = model2.inference(x)

    # 创建一个加载模型文件的对象
    saver = tf.train.Saver()
    # 用来保存图片的id
    test_img_ids = []
    # 用来保存图片的预测结果
    test_pred_labels = []
    # 计算需要迭代的次数
    steps = (test_data.data_size - 1) // batch_size + 1
    # 设置模型文件的路径
    model_path = "../checkpoints/resnet/model_epoch2_0.5000.ckpt"



    with tf.Session() as sess:
        sess.run(test_init_op)
        # 加载模型文件
        saver.restore(sess, model_path)
        for step in range(steps):
            # 获取数据
            image_data, image_id = sess.run(next_batch)
            # 预测图片的标签
            pred_label = sess.run(output_y, feed_dict={x: image_data, is_training: False})
            pred_prob = tf.nn.softmax(pred_label)
            # 保存预测的结果
            test_img_ids.extend(image_id)
            test_pred_labels.extend(np.round(sess.run(pred_prob)[:, 1], decimals=2))
        data = pd.DataFrame({"id": test_img_ids, "pre_label": test_pred_labels,"true_label": true_label})
        data.sort_values(by="id", ascending=True, inplace=True)
        data = data.reset_index(drop=True)  # 重置index

        # 保存结果
        data.to_csv("resnet_test_result.csv", index=False)


    ids = np.array(data.id)
    true_label = np.array(data.true_label)
    pred_labels = data.pre_label

    data_size = len(data)
    # 获取包名
    bag_names = []
    for i in range(data_size):
        bag_name, _ = ids[i].decode('ascii').split('_')
        bag_names.append(bag_name)
    bag_names = np.unique(bag_names)  # 去重
    bag_size = len(bag_names)

    # 存到list中[包名，标签，[pre1,pre2,pre3,pre4,pre5]]
    list = []
    for i in range(bag_size):
        arr = []
        pred_bag_label = []
        for j in range(data_size):
            if bag_names[i] == ids[j].decode('ascii').split('_')[0]:
                k = j
                pred_bag_label.append(pred_labels[j])
        arr.append(bag_names[i])
        arr.append(true_label[k])
        arr.append(pred_bag_label)
        list.append(arr)
    # 计算ACC,TPR,TNR
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(bag_size):
        pre_max = max(list[i][2])
        # print(list[i][0], list[i][1],pre_max)
        if list[i][1] == 1:
            if pre_max > 0.6:
                TP += 1
            else:
                FN += 1
        elif list[i][1] == 0:
            if pre_max < 0.6:
                TN += 1
            else:
                FP += 1
    sum = TP + FN + TN + FP
    ACC = (TP+TN)/sum
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    print("ACC = {},TPR = {}, TNR = {}".format(ACC, TPR, TNR))

# 输出特征向量
def feature_extraction():
    label_name_to_num = {"no_funi": 0, "funi": 1}
    # 获取需要预测结果的所有数据
    img_ids, img_labels, img_paths = get_img_infos("extract_features", MISVM_txt, label_name_to_num)

    test_dataset = pd.DataFrame({"img_id": img_ids, "img_path": img_paths, "img_label": img_labels})
    with tf.device("/cpu:0"):
        test_data = ImageDataGenerator(test_dataset, mode="extract_features", batch_size=batch_size,
                                       num_classes=num_classes)
        iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        next_batch = iterator.get_next()
    # 初始化测试集中的图片数据
    test_init_op = iterator.make_initializer(test_data.data)

    # 获得特征向量
    output_scale5 = model2.inference(x)

    # 特征向量保存路径
    save_txt_dir = "/media/Store/tyh/funi_across_validation/feature"
    # 创建一个加载模型文件的对象
    saver = tf.train.Saver()

    # 用来保存每一个step的包名
    image_bag_name = []
    # 用来保存图片的id
    test_img_ids = []
    # 用来保存图片属于的包名
    test_bag_names = []
    # 用来保存图片的label
    test_img_labels = []
    # 用来保存特征向量
    test_img_features = []
    # 计算需要迭代的次数
    steps = (test_data.data_size - 1) // batch_size + 1

    # 设置模型文件的路径
    model_path = "../checkpoints/resnet/model_epoch20_0.4961.ckpt"
    with tf.Session() as sess:

        sess.run(test_init_op)
        # 加载模型文件
        saver.restore(sess, model_path)
        for step in range(steps):
            # 获取数据
            image_data, image_id, image_label = sess.run(next_batch)

            # print(image_id)
            for i in range(0, len(image_id)):
                image_id[i] = image_id[i].decode('ascii')  # 二进制解码
                bag_name, _ = image_id[i].split('_')  # 获取包名
                # print(image_id[i],bag_name)
                image_bag_name.append(bag_name)

            # 预测图片的标签
            feature = sess.run(output_scale5, feed_dict={x: image_data, is_training: False})

            # pred_prob = tf.nn.softmax(pred_label)

            # 保存预测的结果
            test_img_ids.extend(image_id)
            test_bag_names.extend(image_bag_name)
            test_img_labels.extend(image_label)
            # img_features.extend(np.round(sess.run(feature)[:,1],decimals=2))
            test_img_features.extend(feature)

            image_bag_name = []  # 每个step清空一次image_bag_name，需要新的image_bag_name
        print(len(test_img_ids), len(test_bag_names), len(test_img_labels), len(test_img_features),
              len(test_img_features[0]))

        # data = pd.DataFrame({"id":test_img_ids,"bag":test_bag_names,"label":test_img_labels,"features":test_img_features},columns=['id', 'bag', 'label','features'])

        # data = pd.DataFrame(test_img_features)
        # data.sort_values(by="id",ascending=True,inplace=True)
        # 保存结果
        # data.to_csv("feature_layer7_new.csv",index=False)
        # exit()
        save_txt_path = os.path.join(save_txt_dir, "resnet_feature.txt")
        with open(save_txt_path, mode="w", encoding="utf-8") as f_write:
            for i in range(1364):
                id = test_img_ids[i]
                bag = test_bag_names[i]
                label = test_img_labels[i]
                f_write.write("%s,%s,%s" % (id, bag, label))
                for j in range(2048):
                    f_write.write(",%s" % (test_img_features[i][j]))

                f_write.write("\n")
            f_write.close()


        data = pd.DataFrame({"id": test_img_ids, "bag_names": test_bag_names, "ins_labels": test_img_labels, "img_features": test_img_features},columns = ["id","bag_names","ins_labels","img_features"])
        data.sort_values(by="id", ascending=True, inplace=True)
        data = data.reset_index(drop=True)  # 重置index

        split_dataset_2016.split_feature(data, "../feature/split_data")

if __name__ == "__main__":
    # train()
    # feature_extraction()
    genrate_pre_result()
