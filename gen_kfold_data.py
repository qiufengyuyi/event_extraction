import os
import numpy as np
from data_processing.event_prepare_data import EventTypeClassificationPrepare, EventRolePrepareMRC
from configs.event_config import event_config

if __name__ == "__main__":
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    # bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("bert_config_path"))
    event_type_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("event_type_file"))
    # data_loader =EventTypeClassificationPrepare(vocab_file_path,512,event_type_file)
    # train_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_train"))
    # eval_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_eval"))
    # train_data_list,train_label_list,train_token_type_id_list,dev_data_list,dev_label_list,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,is_train=True)
    slot_file = os.path.join(event_config.get("slot_list_root_path"),
                             event_config.get("bert_slot_complete_file_name_role"))
    schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))
    query_map_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("query_map_file"))

    data_loader = EventRolePrepareMRC(vocab_file_path, 512, slot_file, schema_file, query_map_file)
    train_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_train"))
    eval_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_eval"))
    # data_list,label_start_list,label_end_list,query_len_list,token_type_id_list
    # train_datas, train_labels_start,train_labels_end,train_query_lens,train_token_type_id_list,dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,True)
    # dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._read_json_file(eval_file,None,False)
    # train_datas, train_labels_start,train_labels_end,train_query_lens,train_token_type_id_list,dev_datas, dev_labels_start,dev_labels_end,dev_query_lens,dev_token_type_id_list = data_loader._merge_ee_and_re_datas(train_file,eval_file,"relation_extraction/data/train_data.json","relation_extraction/data/dev_data.json")
    data_loader.k_fold_split_data(train_file, eval_file, True)
    # import numpy as np
    # train_query_lens = np.load("data/fold_data_{}/query_lens_train.npy".format(0),allow_pickle=True)
    # print(train_query_lens[0])

    # re_train_file = "relation_extraction/data/train_data.json"
    # re_dev_file = "relation_extraction/data/dev_data.json"
    # # data_loader.k_fold_split_data(train_file,eval_file,re_train_file,re_dev_file,True,6)
    # train_datas = np.load("data/re15000_neg_fold_data_{}/token_ids_train.npy".format(0),allow_pickle=True)
    # train_labels = np.load("data/re15000_neg_fold_data_{}/multi_labels_train.npy".format(0),allow_pickle=True)
    # train_query_lens = np.load("data/re15000_neg_fold_data_{}/query_lens_train.npy".format(0),allow_pickle=True)
    # train_token_type_id_list = np.load("data/re15000_neg_fold_data_{}/token_type_ids_train.npy".format(0),allow_pickle=True)
    # # dev_datas = np.load("data/re10000_neg_fold_data_{}/token_ids_dev.npy".format(args.fold_index),allow_pickle=True)
    # # dev_labels = np.load("data/re10000_neg_fold_data_{}/multi_labels_dev.npy".format(args.fold_index),allow_pickle=True)
    # # dev_query_lens = np.load("data/re10000_neg_fold_data_{}/query_lens_dev.npy".format(args.fold_index),allow_pickle=True)
    # # dev_token_type_id_list = np.load("data/re10000_neg_fold_data_{}/token_type_ids_dev.npy".format(args.fold_index),allow_pickle=True)
    # for i in range(len(train_datas)):
    #     if len(train_datas[i])!= len(train_token_type_id_list[i]):
    #         print(len(train_datas[i]))
    #         print(len(train_token_type_id_list[i]))
    #         print(train_datas[i])
    #         print(train_token_type_id_list[i])
    # train_datas = np.load("data/verify_neg_fold_data_{}/token_ids_train.npy".format(0),allow_pickle=True)
    # train_has_answer_label_list = np.load("data/verify_neg_fold_data_{}/has_answer_train.npy".format(0),allow_pickle=True)
    # train_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_train.npy".format(0),allow_pickle=True)
    # dev_datas = np.load("data/verify_neg_fold_data_{}/token_ids_dev.npy".format(0),allow_pickle=True)
    # dev_has_answer_label_list = np.load("data/verify_neg_fold_data_{}/has_answer_dev.npy".format(0),allow_pickle=True)
    # dev_token_type_id_list = np.load("data/verify_neg_fold_data_{}/token_type_ids_dev.npy".format(0),allow_pickle=True)
    # train_query_lens = np.load("data/verify_neg_fold_data_{}/query_lens_train.npy".format(0),allow_pickle=True)
    # dev_query_lens = np.load("data/verify_neg_fold_data_{}/query_lens_dev.npy".format(0),allow_pickle=True)
    # train_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_train.npy".format(0),allow_pickle=True)
    # dev_start_labels = np.load("data/verify_neg_fold_data_{}/labels_start_dev.npy".format(0),allow_pickle=True)
    # train_end_labels = np.load("data/verify_neg_fold_data_{}/labels_end_train.npy".format(0),allow_pickle=True)
    # dev_end_labels = np.load("data/verify_neg_fold_data_{}/labels_end_dev.npy".format(0),allow_pickle=True)
    # for i in range(len(dev_has_answer_label_list)):
    #     if sum(dev_start_labels[i]) == 0 and dev_has_answer_label_list[i] != 0:
    #         print(i)
    #         print(dev_start_labels[i])
    #         print(dev_has_answer_label_list[i])
    #     if sum(dev_start_labels[i]) != 0 and dev_has_answer_label_list[i] == 0:
    #         print(i)
    #         print(dev_start_labels[i])
    #         print(dev_has_answer_label_list[i])
