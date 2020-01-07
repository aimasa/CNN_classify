# CNN_classify

## 简介

本项目使用

- python3.5
- TensorFlow1.13

新闻分类，对`房产、彩票、 财经`三类新闻进行分类。

## 数据

使用数据集为`THUCNews`，[THUCNews下载](http://thuctc.thunlp.org/)

因为训练集数据量太大，要训练很久，所以就切分了三个类别出来，并且以6:1的比例摘取。

测试集是对该数据集只有506个文件，每个类别有两百份txt左右。

## 训练

相关依赖包下载：

​    `pip install requirements.txt`

运行文件：
> train.py

参数设置文件：

> normal_param.py

其中`vocab`是存储词对应下标的文件，可以删除再重新生成哦（但是要很久）。

## 准确率

| 准确率 | 损失值 |
| ------ | ------ |
| 96.6%  | 0.99   |



## 参考

详细讲解以及论文地址看博客：

[解读CNN在文本分类方面的应用]([https://aimasa.github.io/2019/11/20/CNN%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/](https://aimasa.github.io/2019/11/20/CNN论文笔记/))

[解读该代码内容]([https://aimasa.github.io/2019/12/25/TensorFlow%E5%AE%9E%E7%8E%B0cnn%E8%B8%A9%E5%9D%91%E7%82%B9/#more](https://aimasa.github.io/2019/12/25/TensorFlow实现cnn踩坑点/#more))