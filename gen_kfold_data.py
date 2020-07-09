import os
import numpy as np
from data_processing.event_prepare_data import EventTypeClassificationPrepare, EventRolePrepareMRC
from configs.event_config import event_config


def gen_type_classification_data():
    """
    generate event type classification data of index_type_fold_data_{}
    """
    # bert vocab file path
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    # bert config file path
    bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("bert_config_path"))
    # event type list file path
    event_type_file =  os.path.join(event_config.get("slot_list_root_path"), event_config.get("event_type_file"))
    data_loader =EventTypeClassificationPrepare(vocab_file_path,512,event_type_file)
    # train file
    train_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_train"))
    # eval file
    eval_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_eval"))
    data_loader.k_fold_split_data(train_file,eval_file,True)

def gen_role_class_data():
    """
    generate role mrc data for verify_neg_fold_data_{}
    """
    # bert vocab file path
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    # event role slot list file path
    slot_file = os.path.join(event_config.get("slot_list_root_path"),event_config.get("bert_slot_complete_file_name_role"))
    # schema file path
    schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))
    # query map file path
    query_file = os.path.join(event_config.get("slot_list_root_path"),event_config.get("query_map_file"))
    data_loader = EventRolePrepareMRC(vocab_file_path,512,slot_file,schema_file,query_file)
    train_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_train"))
    eval_file = os.path.join(event_config.get("data_dir"),event_config.get("event_data_file_eval"))
    data_loader.k_fold_split_data(train_file,eval_file,True)

if __name__ == "__main__":
    gen_type_classification_data()
    gen_role_class_data()
