import jieba
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.contrib import learn

from model import cnn
from pro_data.process_data import process_data
import time
import datetime
import random
import normal_param
# import keras.backend.tensorflow_backend as KTF
import evaluation

print("loading data……")


def train():
    print("loading data……")
    process_data_init = process_data()
    process_data_init.split_data_file(normal_param.train_path)
    # train_data, train_label, dev_data, dev_label, vocal_size_train = process_data_init.deal_data(
    #     normal_param.train_path, n_part=0)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=normal_param.allow_soft_placement,
                                      log_device_placement=normal_param.log_device_placement)
        # session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        # KTF.set_session(sess)
        with sess.as_default():
            cnn_init = cnn.model_cnn(sequence_length=normal_param.sequence_length, num_classes=normal_param.num_class,
                                     vocab_size=normal_param.vocal_size_train,
                                     embedding_size=normal_param.embedding_dim,
                                     filters_size=list(map(int, normal_param.filter_sizes.split(","))),
                                     num_filters=normal_param.num_filters)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn_init.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            time_step = str(int(time.time()))
            print(time.time())
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", normal_param.model_name))
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", cnn_init.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn_init.accuracy)

            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summary", "train")
            train_summary_write = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summary", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())

            max_acc = 0

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoint"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("CNN restore from the checkpoint {0}".format(ckpt))
                # current_step = int(ckpt.split('-')[-1])


            # sess.run(tf.initializers)

            def train_step(x_bratch, y_bratch, writer):

                # print(y_bratch.shape())
                feed_dic = {
                    cnn_init.input_x: x_bratch,
                    cnn_init.input_y: y_bratch,
                    cnn_init.dropout_keep_prob: normal_param.dropout_keep_prob

                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn_init.loss, cnn_init.accuracy], feed_dic)
                # print("scores: ", score)
                time_str = datetime.datetime.now().isoformat()
                print('{}: step {}, loss {:g}, acc {:g}'.format(
                    time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)



            def dev_step(x_bratch, y_bratch, writer=None):
                '''在开发集上验证数据集'''
                feed_dic = {
                    cnn_init.input_x: x_bratch,
                    cnn_init.input_y: y_bratch,
                    cnn_init.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn_init.loss, cnn_init.accuracy], feed_dic)
                time_str = datetime.datetime.now().isoformat()
                print('{}: step {}, loss {:g}, acc {:g}'.format(
                    time_str, step, loss, accuracy))

                if writer:
                    writer.add_summary(summaries, step)

            # batches = process_data_init.batch_iter(list(zip(train_data, train_label)), normal_param.batch_size,
            #                                        normal_param.num_epochs)
            batches = process_data_init.batch_iter(normal_param.batch_size, normal_param.num_epochs)
            for batch, dev_data, dev_label, is_save in batches:
            # for batch, is_save in batches:
                x_batch, y_batch = zip(*batch)
                # print("y_batch", y_batch)
                train_step(x_batch, y_batch, train_summary_write)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % normal_param.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(dev_data, dev_label, writer=dev_summary_writer)


                if current_step % normal_param.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if is_save:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    train()
