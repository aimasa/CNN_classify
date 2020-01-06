import tensorflow as tf
import numpy as np
import re
import os
from pro_data import process_data as process
import normal_param
from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from tqdm import tqdm

model_graph_dir = './runs/' + normal_param.model_name + '/checkpoint/model-3200.meta'  # 模型中.meta文件


# dict = {"财经":2, "彩票":0, "房产":1}
def pro_data():
    '''读取验证集中数据
    :return dev_data 验证集数据列表
    :return dev_label 验证集列表中数据对应label
    '''
    pro = process.process_data()
    pro.split_data_file(normal_param.dev_path)
    dev_data, dev_label, _ = pro.deal_data(part=len(pro.all_text_path), n_part=0)
    return dev_data, dev_label


# def get_array(texts_content):
#     # list_arr_text = self.deal_text(x_texts)
#     x_texts = [x for x, label in texts_content]
#     x_labels = np.array([label for x, label in texts_content])
#     vocab_processor = learn.preprocessing.VocabularyProcessor(normal_param.max_document_length)
#     x = np.array(list(vocab_processor.fit_transform(x_texts)))
#     shuffle_indices = np.random.permutation(np.arange(len(x_labels)))
#     text_shuffled = x[shuffle_indices]
#     label_shuffled = x_labels[shuffle_indices]
#
#     print(label_shuffled)
#     pro = process.process_data()
#     return text_shuffled, pro.dense_to_one_hot(label_shuffled, normal_param.num_class)


def test(data_array, one_hot_labels):
    '''对输入有对应label的验证集进行对应模型准确度验证、
    :param data_array 验证集文字对应下标id组成数组
    :param one_hot_labels data_array的对应label
    :return pred 验证集对应预测类别
    :return accuracy 验证集预测准确度

    '''
    with tf.Session() as sess:
        # 这里需要判断一下模型所在文件夹
        saver = tf.train.import_meta_graph(model_graph_dir)
        saver.restore(sess, tf.train.latest_checkpoint(
            os.path.join(os.path.curdir, "runs", normal_param.model_name, "checkpoint")))  # .data文件
        # pred = tf.get_collection('predictions')[0]

        graph = tf.get_default_graph()
        # var_list = [v.name for v in tf.global_variables()]
        # print(var_list)
        # for names in graph._names_in_use:
        #     print(names)
        # embeding = graph.get_tensor_by_name("embedding/W:0")
        input_y = graph.get_tensor_by_name('input_y:0')
        input_x = graph.get_tensor_by_name('input_x:0')
        pred = graph.get_tensor_by_name("output/predictions:0")
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        # score = graph.get_tensor_by_name("output/scores:0")
        # input_y = graph.get_operation_by_name('input_y').outputs[0]
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
        # W = graph.get_tensor_by_name('output/W:0')

        pred, accuracy = sess.run([pred, accuracy],
                                  feed_dict={input_x: data_array, input_y: one_hot_labels, dropout_keep_prob: 1.0})

    print('Successfully load the pre-trained model!')
    return pred, accuracy


def read_files(path):
    files_name = os.listdir(path)
    files_path = [os.path.join(path, file_name) for file_name in files_name]
    file_label = [file_name for file_name in files_name]
    print(files_path)
    return files_path, file_label


# def run(path):
#     '''读取path下的所有文件
#     :return dev_data 验证集数据列表
#     :return dev_label 验证集列表中数据对应label
#     '''
def run():
    '''读取验证集中数据
    :return dev_data 验证集数据列表
    :return dev_label 验证集列表中数据对应label
    '''
    # texts_path, texts_label = read_files(path)
    dev_data, dev_label = pro_data()

    return dev_data, dev_label
    # 如果单独拿验证集得到预测结果的话，可以把return这句注释掉，然后把下面100-102三行解除注释，运行run
    # pro, accuracy = test(dev_data, dev_label)
    # print(pro)
    # print("this is acc", accuracy)


if __name__ == "__main__":
    path = "F:/实验数据暂存/tree_test"
    # path = "F:/实验数据暂存/tree/test-fangcan.txt"
    # path = "F:/实验数据暂存/tree/test-caipiao.txt"
    run()
