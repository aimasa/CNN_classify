import tensorflow as tf


class model_cnn(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filters_size, num_filters, l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filters_size = filters_size
        self.num_filters = num_filters
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.channel_num = 1
        self.l2_reg_lambda = l2_reg_lambda
        self.count = 1

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W")
            # 随机化W，得到维度是self.vocab_size个self.embedding_size大小的矩阵，随机值在-1.0-1.0之间
            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 从id(索引)找到对应的One-hot encoding
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)
            # 增加维度

        pooled_outputs = []
        for i, filter_size in enumerate(self.filters_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, self.channel_num, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")  # 权重
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")  # 偏移差
                conv = tf.nn.conv2d(self.embedding_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")
                # 最大值池化
                pooled_outputs.append(pooled)
        self.num_filters_total = self.num_filters * len(self.filters_size)
        # 卷积的总数量
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        #     全连接层dropout防拟合

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.num_filters_total, self.num_classes]), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            # if l2_reg_lambda:
            #     W_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(W)
            #     tf.add_to_collection("losses", W_l2_loss)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # (self.h_drop * W + b)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # 找出分数中的最大值，就是预测值

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(self.input_y, self.scores)
            # softmax_cross_entropy_with_logits_v2第一个是对应标签的labels值，第二个logits值是预测分数，会自动转为对应的标签值进行计算
            mse_loss = tf.reduce_mean(losses, -1)
            tf.add_to_collection("losses", mse_loss)
            self.loss = tf.add_n(tf.get_collection("losses"))
            self.loss = mse_loss

        with tf.name_scope("accuracy"):
            correct_prtdictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prtdictions, "float"), name="accuracy")
