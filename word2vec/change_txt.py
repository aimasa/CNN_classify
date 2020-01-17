import gensim
import numpy as np
import re
from word2vec import MyCorpus as word2vec
from gensim.models.word2vec import Word2Vec
import os
import pickle

def input_to_list(lists, mode="split"):
    '''
    将输入列表进行数据整理，并且返回整理好的列表
    :param lists: 需要被删除.txt以及\n后缀的列表
    :param mode: split 根据"_"分割字符串，no split 不根据"_"分割字符串
    :return: all_vocab 处理好后的列表
    '''
    all_vocab = []
    for list in lists:
        list = clean_stop(list)
        if mode == "no split":
            all_vocab += [list]
        elif mode == "split":
            if re.search("_", list):
                for str in list.split("_"):
                    all_vocab += [str]
            else:
                all_vocab += [list]
    return all_vocab


def clean_stop(str):
    '''
    删去停用词".txt \n"
    :param str:需要被处理的字符串
    :return: 被处理好的字符串
    '''
    str = re.sub("\.txt\\n", "", str)
    return str


def deal_text(list_text):
    '''
    对数据进行处理，将list_text里面的词通过word2vec训练生成模型（没有保存的那种，我就懒得做容错处理了）
    :param list_text:（list）词列表(如：['a','b'],每个词汇都是独立的个体，没有其他特殊符号)
    :return: 通过word2vec训练生成的模型
    '''
    # vocab = [s.encode('utf-8').split() for s in list_text]
    raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]

    # 切分词汇
    sentences = [s.split() for s in list_text]
    model = Word2Vec(sentences=sentences, min_count=0, size=300)

    model.init_sims(replace=True)
    return model
    # word_to_vec(list_text, model)
    # return list_arr_text


def word_to_vec(list_txt, model):
    '''
    将词汇列表中的词汇转换为array形式。其中词汇中含有"_"的则分割开来，求取其分割开后的词汇的array的平均值
    :param list_txt: (list) 需要被转换的词汇列表(eg: ['a','b','a_b']只允许有"_"此种特殊符号)
    :param model: 训练好的词转换模型
    :return: 词转换成array列表并且存储为pk格式的文件。
    '''
    all_vec = []
    for index, list in enumerate(list_txt):
        if re.search("_", list):
            list_vec = []
            for str in list.split("_"):
                list_vec += [model.wv[str]]
            vec = np.mean(np.array(list_vec), axis=0)
        else:
            vec = model.wv[list]
        all_vec += [vec]
        pickle.dump(all_vec, open("./english_txt_name.pk", "wb"))



def read_txt(path):
    '''
    一次性读取txt文本内所有内容
    :param path: txt文本存储路径
    :return: path路径下对应的文本内容
    '''
    with open(path, "r") as f:
        str = f.readlines()
        print(str)
    return str


def run(path):
    '''
    将txt中的文本除去后缀".txt"转换为词向量，其中，将带有"_"字符串分割开来分别求取词向量，最后得到平均值作为该字符串词向量。
    :param path: txt文本对应的路径
    :return: 词转换成array列表并且存储为pk格式的文件。
    '''
    str_list = []
    txt_content = read_txt(path)
    model = deal_text(input_to_list(txt_content))
    word_to_vec(input_to_list(txt_content, mode="no split"), model)


if __name__ == "__main__":
    run(os.path.join(os.path.curdir, "temp.txt"))
