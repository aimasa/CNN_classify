num_class = 3
embedding_dim = 128
allow_soft_placement = True
log_device_placement = False
# train_path = "F:/实验数据暂存/THUCNews/THUCNews"
train_path = "F:/实验数据暂存/tempNews"
dev_path = "F:/实验数据暂存/tree"
# tst_path = "F:/实验数据暂存/tempNew"
n = ""
batch_size = 64
num_epochs = 50
filter_sizes = '2, 3, 4'
num_filters = 64
dropout_keep_prob = 0.5
evaluate_every = 100
checkpoint_every = 100
max_document_length = 500
num_database = 1
stop_words_path = "F:/phython workspace/stopwords/中文停用词表.txt"
cache_root_dir = './cache'
dev_sample_index = 0.1
vocal_size_train = 290000
sequence_length = 500
l2_reg_lambda = 0.0
model_name = "run_10"
# model_name = "run_9_op_0.001_batch_size_128"
num_size = 1
vocab_dir = "./vocab"