import os
import numpy as np
import time
import logging
from common_utils import set_logger
import tensorflow as tf
from sklearn.metrics import f1_score

from models.bert_mrc import bert_mrc_model_fn_builder

from models.bert_event_type_classification import bert_classification_model_fn_builder
from data_processing.data_utils import *
from data_processing.event_prepare_data import EventRolePrepareMRC, EventTypeClassificationPrepare
# from data_processing.event_prepare_data import EventRoleClassificationPrepare
from data_processing.event_prepare_data import event_input_bert_mrc_mul_fn, event_index_class_input_bert_fn, event_input_bert_mrc_fn
from data_processing.event_prepare_data import event_binclass_input_bert_fn
from models.bert_event_type_classification import bert_binaryclassification_model_fn_builder
from data_processing.event_prepare_data import event_input_verfify_mrc_fn
from models.event_verify_av import event_verify_mrc_model_fn_builder
from configs.event_config import event_config

# import horovod.tensorflow as hvd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = set_logger("[run training]")


# logger = logging.getLogger('train')
# logger.setLevel(logging.INFO)
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    words_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words_seq')
    receiver_tensors = {'words': words, 'text_length': nwords, 'words_seq': words_seq}
    features = {'words': words, 'text_length': nwords, 'words_seq': words_seq}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def bert_serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    receiver_tensors = {'words': words, 'text_length': nwords}
    features = {'words': words, 'text_length': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def bert_event_type_serving_input_receiver_fn():
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
    type_index_ids = tf.placeholder(dtype=tf.int32, shape=[None, 65], name="type_index_in_ids_list")
    receiver_tensors = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids,
                        'type_index_in_ids_list': type_index_ids}
    features = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids,
                'type_index_in_ids_list': type_index_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def bert_event_bin_serving_input_receiver_fn():
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
    receiver_tensors = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids}
    features = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def bert_mrc_serving_input_receiver_fn():
    # features['words'],features['text_length'],features['query_length'],features['token_type_ids']
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    query_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="query_length")
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
    receiver_tensors = {'words': words, 'text_length': nwords, 'query_length': query_lengths,
                        'token_type_ids': token_type_ids}
    features = {'words': words, 'text_length': nwords, 'query_length': query_lengths, 'token_type_ids': token_type_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def run_event_role_mrc(args):
    """
    baseline 用mrc来做事件role抽取
    :param args:
    :return:
    """
    model_base_dir = event_config.get(args.model_checkpoint_dir).format(args.fold_index)
    pb_model_dir = event_config.get(args.model_pb_dir).format(args.fold_index)
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"),
                                    event_config.get("bert_config_path"))
    slot_file = os.path.join(event_config.get("slot_list_root_path"),
                             event_config.get("bert_slot_complete_file_name_role"))
    schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))
    query_map_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("query_map_file"))
    data_loader = EventRolePrepareMRC(vocab_file_path, 512, slot_file, schema_file, query_map_file)
    # train_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_train"))
    # eval_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_eval"))
    # data_list,label_start_list,label_end_list,query_len_list,token_type_id_list
    # train_datas, train_labels_start,train_labels_end,train_query_lens,train_token_type_id_list,dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,True)
    # dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._read_json_file(eval_file,None,False)
    # train_datas, train_labels_start,train_labels_end,train_query_lens,train_token_type_id_list,dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._merge_ee_and_re_datas(train_file,eval_file,"relation_extraction/data/train_data.json","relation_extraction/data/dev_data.json")

    train_datas = np.load("data/verify_neg_fold_data_{}/token_ids_train.npy".format(args.fold_index), allow_pickle=True)
    # train_labels = np.load("data/verify_neg_fold_data_{}/multi_labels_train.npy".format(args.fold_index), allow_pickle=True)
    train_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_train.npy".format(args.fold_index), allow_pickle=True)
    train_end_labels = np.load("data/verify_neg_fold_data_{}/labels_end_train.npy".format(args.fold_index), allow_pickle=True)
    train_query_lens = np.load("data/verify_neg_fold_data_{}/query_lens_train.npy".format(args.fold_index), allow_pickle=True)
    train_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_train.npy".format(args.fold_index),
                                       allow_pickle=True)
    dev_datas = np.load("data/verify_neg_fold_data_{}/token_ids_dev.npy".format(args.fold_index), allow_pickle=True)
    dev_labels = np.load("data/verify_neg_fold_data_{}/multi_labels_dev.npy".format(args.fold_index), allow_pickle=True)
    dev_query_lens = np.load("data/verify_neg_fold_data_{}/query_lens_dev.npy".format(args.fold_index), allow_pickle=True)
    dev_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_dev.npy".format(args.fold_index),
                                     allow_pickle=True)

    train_samples_nums = len(train_datas)
    dev_samples_nums = len(dev_datas)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****dev_set sample nums:{}'.format(dev_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    # train_steps_nums = each_epoch_steps * args.epochs // hvd.size()
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch * each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.labels_map_len,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps": decay_steps, "train_steps": train_steps_nums,
              "num_warmup_steps": int(train_steps_nums * 0.1)}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=3,
        # train_distribute=dist_strategy

    )
    bert_init_checkpoints = os.path.join(event_config.get("bert_pretrained_model_path"),
                                         event_config.get("bert_init_checkpoints"))
    # init_checkpoints = "output/model/merge_usingtype_roberta_traindev_event_role_bert_mrc_model_desmodified_lowercase/checkpoint/model.ckpt-1218868"
    model_fn = bert_mrc_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)
    if args.do_train:
        # train_input_fn = lambda: event_input_bert_mrc_mul_fn(
        #     train_datas, train_labels, train_token_type_id_list, train_query_lens,
        #     is_training=True, is_testing=False, args=args)
        train_input_fn = lambda: event_input_bert_mrc_fn(
            train_datas, train_start_labels, train_end_labels, train_token_type_id_list, train_query_lens,
            is_training=True, is_testing=False, args=args)
        # eval_input_fn = lambda: event_input_bert_mrc_mul_fn(
        #     dev_datas, dev_labels, dev_token_type_id_list, dev_query_lens,
        #     is_training=False, is_testing=False, args=args)
        eval_input_fn = lambda: event_input_bert_mrc_fn(
            train_datas, train_start_labels, train_end_labels, train_token_type_id_list, train_query_lens,
            is_training=True, is_testing=False, args=args)
        
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_mrc_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter], throttle_secs=0)
        # for _ in range(args.epochs):

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # "bert_ce_model_pb"
        estimator.export_saved_model(pb_model_dir, bert_mrc_serving_input_receiver_fn)


def run_event_classification(args):
    """
    事件类型分析，多标签二分类问题，借鉴NL2SQL预测column的方法
    :param args:
    :return:
    """
    model_base_dir = event_config.get(args.model_checkpoint_dir).format(args.fold_index)
    pb_model_dir = event_config.get(args.model_pb_dir).format(args.fold_index)
    print(model_base_dir)
    print(pb_model_dir)
    
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"),
                                    event_config.get("bert_config_path"))
    event_type_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("event_type_file"))
    data_loader = EventTypeClassificationPrepare(vocab_file_path, 512, event_type_file)
    train_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_train"))
    eval_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_eval"))
    # train_data_list,train_label_list,train_token_type_id_list,dev_data_list,dev_label_list,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,is_train=True)
    train_data_list = np.load("data/index_type_fold_data_{}/token_ids_train.npy".format(args.fold_index),
                              allow_pickle=True)
    train_label_list = np.load("data/index_type_fold_data_{}/labels_train.npy".format(args.fold_index),
                               allow_pickle=True)
    train_token_type_id_list = np.load("data/index_type_fold_data_{}/token_type_ids_train.npy".format(args.fold_index),
                                       allow_pickle=True)
    train_type_index_ids_list = np.load(
        "data/index_type_fold_data_{}/type_index_in_token_ids_train.npy".format(args.fold_index), allow_pickle=True)
    dev_data_list = np.load("data/index_type_fold_data_{}/token_ids_dev.npy".format(args.fold_index), allow_pickle=True)
    dev_label_list = np.load("data/index_type_fold_data_{}/labels_dev.npy".format(args.fold_index), allow_pickle=True)
    dev_token_type_id_list = np.load("data/index_type_fold_data_{}/token_type_ids_dev.npy".format(args.fold_index),
                                     allow_pickle=True)
    dev_type_index_ids_list = np.load(
        "data/index_type_fold_data_{}/type_index_in_token_ids_dev.npy".format(args.fold_index), allow_pickle=True)
    train_labels = np.array(train_label_list)
    # print(train_labels.shape)
    print(train_labels.shape)
    a = np.sum(train_labels, axis=0)
    a = [max(a) / ele for ele in a]
    class_weight = np.array(a)
    class_weight = np.reshape(class_weight, (1, 65))
    print(class_weight)
    # dev_datas,dev_token_type_ids,dev_labels = data_loader._read_json_file(eval_file)
    train_samples_nums = len(train_data_list)
    dev_samples_nums = len(dev_data_list)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    # train_steps_nums = each_epoch_steps * args.epochs // hvd.size()
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch * each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.labels_map_len,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps": decay_steps, "class_weight": class_weight}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    # "bert_ce_model_dir"
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # config_tf.gpu_options.visible_device_list = str(hvd.local_rank())
    # checkpoint_path = os.path.join(bert_config.get(args.model_checkpoint_dir), str(hvd.rank()))

    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=train_steps_nums + 10,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=1,
        # train_distribute=dist_strategy

    )
    bert_init_checkpoints = os.path.join(event_config.get("bert_pretrained_model_path"),
                                         event_config.get("bert_init_checkpoints"))
    model_fn = bert_classification_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train:
        # train_input_fn = lambda: data_loader.create_dataset(is_training=True,is_testing=False, args=args)
        # eval_input_fn = lambda: data_loader.create_dataset(is_training=False,is_testing=False,args=args)
        # train_X,train_Y = np.load(data_loader.train_X_path,allow_pickle=True),np.load(data_loader.train_Y_path,allow_pickle=True)

        # train_input_fn = lambda :event_class_input_bert_fn(train_data_list,token_type_ids=train_token_type_id_list,label_map_len=data_loader.labels_map_len,
        #                                                  is_training=True,is_testing=False,args=args,input_Ys=train_label_list)

        train_input_fn = lambda: event_index_class_input_bert_fn(train_data_list,
                                                                 token_type_ids=train_token_type_id_list,
                                                                 type_index_ids_list=train_type_index_ids_list,
                                                                 label_map_len=data_loader.labels_map_len,
                                                                 is_training=True, is_testing=False, args=args,
                                                                 input_Ys=train_label_list)
        # eval_X,eval_Y = np.load(data_loader.valid_X_path,allow_pickle=True),np.load(data_loader.valid_Y_path,allow_pickle=True)

        # eval_input_fn = lambda: event_class_input_bert_fn(dev_data_list,token_type_ids=dev_token_type_id_list,label_map_len=data_loader.labels_map_len,
        #                                                 is_training=False,is_testing=False,args=args,input_Ys=dev_label_list)
        eval_input_fn = lambda: event_index_class_input_bert_fn(dev_data_list, token_type_ids=dev_token_type_id_list,
                                                                type_index_ids_list=dev_type_index_ids_list,
                                                                label_map_len=data_loader.labels_map_len,
                                                                is_training=False, is_testing=False, args=args,
                                                                input_Ys=dev_label_list)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_event_type_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0, exporters=[exporter])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # "bert_ce_model_pb"
        estimator.export_saved_model(pb_model_dir, bert_event_type_serving_input_receiver_fn)


def run_event_binclassification(args): # 粗读粗读原文，判断问题是否可以在原文找到答案
    """
    retroreader中的eav模块，即第一遍阅读模块，预测该问题是否有回答
    :param args:
    :return:
    """
    model_base_dir = event_config.get(args.model_checkpoint_dir).format(args.fold_index)
    pb_model_dir = event_config.get(args.model_pb_dir).format(args.fold_index)
    print(model_base_dir)
    print(pb_model_dir)
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"),
                                    event_config.get("bert_config_path"))
    event_type_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("event_type_file"))
    # data_loader =EventTypeClassificationPrepare(vocab_file_path,512,event_type_file)
    # train_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_train"))
    # eval_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_eval"))
    # train_data_list,train_label_list,train_token_type_id_list,dev_data_list,dev_label_list,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,is_train=True)
    train_data_list = np.load("data/verify_neg_fold_data_{}/token_ids_train.npy".format(args.fold_index),
                              allow_pickle=True)
    # train_label_list = np.load("data/verify_neg_fold_data_{}/has_answer_train.npy".format(args.fold_index),allow_pickle=True)
    train_label_list = []
    train_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_train.npy".format(args.fold_index),
                                 allow_pickle=True)
    dev_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_dev.npy".format(args.fold_index),
                               allow_pickle=True)

    train_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_train.npy".format(args.fold_index),
                                       allow_pickle=True)
    dev_data_list = np.load("data/verify_neg_fold_data_{}/token_ids_dev.npy".format(args.fold_index), allow_pickle=True)
    # dev_label_list = np.load("data/verify_neg_fold_data_{}/has_answer_dev.npy".format(args.fold_index),allow_pickle=True)
    dev_label_list = []
    dev_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_dev.npy".format(args.fold_index),
                                     allow_pickle=True)

    # dev_datas,dev_token_type_ids,dev_labels = data_loader._read_json_file(eval_file)
    train_samples_nums = len(train_data_list)
    for i in range(train_samples_nums):
        if sum(train_start_labels[i]) == 0:
            train_label_list.append(0)
        else:
            train_label_list.append(1)
    train_label_list = np.array(train_label_list).reshape((train_samples_nums, 1))
    dev_samples_nums = len(dev_data_list)
    for i in range(dev_samples_nums):
        if sum(dev_start_labels[i]) == 0:
            dev_label_list.append(0)
        else:
            dev_label_list.append(1)
    dev_label_list = np.array(dev_label_list).reshape((dev_samples_nums, 1))
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    # train_steps_nums = each_epoch_steps * args.epochs // hvd.size()
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch * each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": 1,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps": decay_steps, "class_weight": 1}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    # "bert_ce_model_dir"
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # config_tf.gpu_options.visible_device_list = str(hvd.local_rank())
    # checkpoint_path = os.path.join(bert_config.get(args.model_checkpoint_dir), str(hvd.rank()))

    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=train_steps_nums + 10,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=1,
        # train_distribute=dist_strategy

    )
    bert_init_checkpoints = os.path.join(event_config.get("bert_pretrained_model_path"),
                                         event_config.get("bert_init_checkpoints"))
    model_fn = bert_binaryclassification_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train:
        # train_input_fn = lambda: data_loader.create_dataset(is_training=True,is_testing=False, args=args)
        # eval_input_fn = lambda: data_loader.create_dataset(is_training=False,is_testing=False,args=args)
        # train_X,train_Y = np.load(data_loader.train_X_path,allow_pickle=True),np.load(data_loader.train_Y_path,allow_pickle=True)

        # train_input_fn = lambda :event_class_input_bert_fn(train_data_list,token_type_ids=train_token_type_id_list,label_map_len=data_loader.labels_map_len,
        #                                                  is_training=True,is_testing=False,args=args,input_Ys=train_label_list)

        train_input_fn = lambda: event_binclass_input_bert_fn(train_data_list, token_type_ids=train_token_type_id_list,
                                                              label_map_len=1,
                                                              is_training=True, is_testing=False, args=args,
                                                              input_Ys=train_label_list)
        # eval_X,eval_Y = np.load(data_loader.valid_X_path,allow_pickle=True),np.load(data_loader.valid_Y_path,allow_pickle=True)

        # eval_input_fn = lambda: event_class_input_bert_fn(dev_data_list,token_type_ids=dev_token_type_id_list,label_map_len=data_loader.labels_map_len,
        #                                                 is_training=False,is_testing=False,args=args,input_Ys=dev_label_list)
        eval_input_fn = lambda: event_binclass_input_bert_fn(dev_data_list, token_type_ids=dev_token_type_id_list,
                                                             label_map_len=1,
                                                             is_training=False, is_testing=False, args=args,
                                                             input_Ys=dev_label_list)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_event_bin_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0, exporters=[exporter])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # "bert_ce_model_pb"
        estimator.export_saved_model(pb_model_dir, bert_event_bin_serving_input_receiver_fn)


def run_event_verify_role_mrc(args):
    """
    retro reader 第二阶段的精度模块，同时训练两个任务，role抽取和问题是否可以回答
    :param args:
    :return:
    """
    model_base_dir = event_config.get(args.model_checkpoint_dir).format(args.fold_index)
    pb_model_dir = event_config.get(args.model_pb_dir).format(args.fold_index)
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"),
                                    event_config.get("bert_config_path"))
    slot_file = os.path.join(event_config.get("slot_list_root_path"),
                             event_config.get("bert_slot_complete_file_name_role"))
    schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))
    query_map_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("query_map_file"))
    data_loader = EventRolePrepareMRC(vocab_file_path, 512, slot_file, schema_file, query_map_file)
    # train_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_train"))
    # eval_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_eval"))
    # data_list,label_start_list,label_end_list,query_len_list,token_type_id_list
    # train_datas, train_labels_start,train_labels_end,train_query_lens,train_token_type_id_list,dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,True)
    # dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._read_json_file(eval_file,None,False)
    # train_datas, train_labels_start,train_labels_end,train_query_lens,train_token_type_id_list,dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._merge_ee_and_re_datas(train_file,eval_file,"relation_extraction/data/train_data.json","relation_extraction/data/dev_data.json")
    train_has_answer_label_list = []
    dev_has_answer_label_list = []
    train_datas = np.load("data/verify_neg_fold_data_{}/token_ids_train.npy".format(args.fold_index), allow_pickle=True)
    # train_has_answer_label_list = np.load("data/verify_neg_fold_data_{}/has_answer_train.npy".format(args.fold_index),allow_pickle=True)
    train_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_train.npy".format(args.fold_index),
                                       allow_pickle=True)
    dev_datas = np.load("data/verify_neg_fold_data_{}/token_ids_dev.npy".format(args.fold_index), allow_pickle=True)
    # dev_has_answer_label_list = np.load("data/verify_neg_fold_data_{}/has_answer_dev.npy".format(args.fold_index),allow_pickle=True)
    dev_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_dev.npy".format(args.fold_index),
                                     allow_pickle=True)
    train_query_lens = np.load("data/verify_neg_fold_data_{}/query_lens_train.npy".format(args.fold_index),
                               allow_pickle=True)
    dev_query_lens = np.load("data/verify_neg_fold_data_{}/query_lens_dev.npy".format(args.fold_index),
                             allow_pickle=True)
    train_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_train.npy".format(args.fold_index),
                                 allow_pickle=True)
    dev_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_dev.npy".format(args.fold_index),
                               allow_pickle=True)
    train_end_labels = np.load("data/verify_neg_fold_data_{}/labels_end_train.npy".format(args.fold_index),
                               allow_pickle=True)
    dev_end_labels = np.load("data/verify_neg_fold_data_{}/labels_end_dev.npy".format(args.fold_index),
                             allow_pickle=True)
    train_samples_nums = len(train_datas)
    for i in range(train_samples_nums):
        if sum(train_start_labels[i]) == 0:
            train_has_answer_label_list.append(0)
        else:
            train_has_answer_label_list.append(1)

    train_has_answer_label_list = np.array(train_has_answer_label_list).reshape((train_samples_nums, 1))
    dev_samples_nums = len(dev_datas)
    for i in range(dev_samples_nums):
        if sum(dev_start_labels[i]) == 0:
            dev_has_answer_label_list.append(0)
        else:
            dev_has_answer_label_list.append(1)
    dev_has_answer_label_list = np.array(dev_has_answer_label_list).reshape((dev_samples_nums, 1))

    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****dev_set sample nums:{}'.format(dev_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    # train_steps_nums = each_epoch_steps * args.epochs // hvd.size()
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch * each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": 2,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps": decay_steps, "train_steps": train_steps_nums,
              "num_warmup_steps": int(train_steps_nums * 0.1)}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=3,
        # train_distribute=dist_strategy

    )
    bert_init_checkpoints = os.path.join(event_config.get("bert_pretrained_model_path"),
                                         event_config.get("bert_init_checkpoints"))
    # init_checkpoints = "output/model/merge_usingtype_roberta_traindev_event_role_bert_mrc_model_desmodified_lowercase/checkpoint/model.ckpt-1218868"
    model_fn = event_verify_mrc_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)
    if args.do_train:
        train_input_fn = lambda: event_input_verfify_mrc_fn(
            train_datas, train_start_labels, train_end_labels, train_token_type_id_list, train_query_lens,
            train_has_answer_label_list,
            is_training=True, is_testing=False, args=args)
        eval_input_fn = lambda: event_input_verfify_mrc_fn(
            dev_datas, dev_start_labels, dev_end_labels, dev_token_type_id_list, dev_query_lens,
            dev_has_answer_label_list,
            is_training=False, is_testing=False, args=args)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_mrc_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter], throttle_secs=0)
        # for _ in range(args.epochs):

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # "bert_ce_model_pb"
        estimator.export_saved_model(pb_model_dir, bert_mrc_serving_input_receiver_fn)
