import tensorflow as tf
import re
import os
import json
import numpy as np
import codecs
from configs.event_config import event_config
from data_processing.event_prepare_data import EventRolePrepareMRC, EventTypeClassificationPrepare
from tensorflow.contrib import predictor

from pathlib import Path

from argparse import ArgumentParser
import datetime
import ipdb;

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class fastPredictTypeClassification:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.data_loader = self.init_data_loader(config)
        self.predict_fn = None
        self.config = config

    def load_models(self):
        subdirs = [x for x in Path(self.model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def load_models_kfold(self, model_path):
        subdirs = [x for x in Path(model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def init_data_loader(self, config):
        event_type_file = os.path.join(config.get("slot_list_root_path"), config.get("event_type_file"))
        # event_type_file = "data/slot_pattern/vocab_all_role_label_noBI_map.txt"

        vocab_file_path = os.path.join(config.get(
            "bert_pretrained_model_path"), config.get("vocab_file"))
        data_loader = EventTypeClassificationPrepare(vocab_file_path, 512, event_type_file)
        # data_loader = EventRoleClassificationPrepare(
        #     vocab_file_path, 512, event_type_file)
        return data_loader

    def parse_test_json(self, test_file):
        id_list = []
        text_list = []
        with codecs.open(test_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                d_json = json.loads(line)
                id_list.append(d_json.get("id"))
                text_list.append(d_json.get("text"))
        return id_list, text_list

    def predict_single_sample(self, text):
        words_input, token_type_ids = self.data_loader.trans_single_data_for_test(
            text)
        predictions = self.predict_fn({'words': [words_input], 'text_length': [
            len(words_input)], 'token_type_ids': [token_type_ids]})
        label = predictions["output"][0]
        return np.argwhere(label > 0.5)

    def predict_single_sample_prob(self, predict_fn, text):
        words_input, token_type_ids, type_index_in_token_ids = self.data_loader.trans_single_data_for_test(
            text)
        predictions = predict_fn({'words': [words_input], 'text_length': [
            len(words_input)], 'token_type_ids': [token_type_ids], 'type_index_in_ids_list': [type_index_in_token_ids]})
        label = predictions["output"][0]
        return label

    def predict_for_all_prob(self, predict_fn, text_list):
        event_type_prob = []
        for text in text_list:
            prob_output = self.predict_single_sample_prob(predict_fn, text)
            event_type_prob.append(prob_output)
        return event_type_prob

    def predict_for_all(self, text_list):
        event_result_list = []
        for text in text_list:
            label_output = self.predict_single_sample(text)
            event_cur_type_list = [self.data_loader.id2labels_map.get(
                ele[0]) for ele in label_output]
            event_result_list.append(event_cur_type_list)
        return event_result_list


class fastPredictCls:
    def __init__(self, model_path, config, query_map_file):
        self.model_path = model_path
        self.data_loader = self.init_data_loader(config, query_map_file)
        self.predict_fn = None
        self.config = config

    def load_models_kfold(self, model_path):
        subdirs = [x for x in Path(model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def init_data_loader(self, config, query_map_file):
        vocab_file_path = os.path.join(config.get(
            "bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(event_config.get("slot_list_root_path"),
                                 event_config.get("bert_slot_complete_file_name_role"))
        schema_file = os.path.join(event_config.get(
            "data_dir"), event_config.get("event_schema"))
        # query_map_file = os.path.join(event_config.get(
        #         "slot_list_root_path"), event_config.get("query_map_file"))
        data_loader = EventRolePrepareMRC(
            vocab_file_path, 512, slot_file, schema_file, query_map_file)
        return data_loader

    def parse_test_json(self, test_file):
        id_list = []
        text_list = []
        with codecs.open(test_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                d_json = json.loads(line)
                id_list.append(d_json.get("id"))
                text_list.append(d_json.get("text"))
        return id_list, text_list

    def predict_single_sample_prob(self, predict_fn, text, query_len, token_type_ids):
        # words_input, token_type_ids,type_index_in_token_ids = self.data_loader.trans_single_data_for_test(
        #     text)
        text_length = len(text)
        predictions = predict_fn({'words': [text], 'text_length': [text_length],
                                  'token_type_ids': [token_type_ids]})
        label_prob = predictions["output"][0]
        return label_prob

    def predict_single_sample_av_prob(self, predict_fn, text, query_len, token_type_ids):
        text_length = len(text)
        predictions = predict_fn({'words': [text], 'text_length': [text_length], 'query_length': [query_len],
                                  'token_type_ids': [token_type_ids]})
        start_ids, end_ids, start_probs, end_probs, has_answer_probs = predictions.get("start_ids"), predictions.get(
            "end_ids"), predictions.get("start_probs"), predictions.get("end_probs"), predictions.get(
            "has_answer_probs")
        return start_ids[0], end_ids[0], start_probs[0], end_probs[0], has_answer_probs[0]

    # def predict_for_all_prob(self,predict_fn,text_list):
    #     event_type_prob = []
    #     for text in text_list:
    #         prob_output = self.predict_single_sample_prob(predict_fn,text)
    #         event_type_prob.append(prob_output)
    #     return event_type_prob
    def extract_entity_from_start_end_ids(self, text, start_ids, end_ids, token_mapping):
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        start_end_tuple_list = []
        # text_cur_index = 0
        for i, start_id in enumerate(start_ids):
            if start_id == 0:
                # text_cur_index += len(token_mapping[i])
                continue
            if end_ids[i] == 1:
                # start and end is the same
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                # entity_str = text[text_cur_index:text_cur_index+cur_entity_len]
                entity_list.append(entity_str)
                start_end_tuple_list.append((i, i))
                # text_cur_index += len(token_mapping[i])
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids):
                # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
                if start_ids[j] == 1:
                    break
                if end_ids[j] == 1:
                    entity_str_index_list = []
                    for index in range(i, j + 1):
                        entity_str_index_list.extend(token_mapping[index])
                    start_end_tuple_list.append((i, j))
                    entity_str = "".join([text[char_index]
                                          for char_index in entity_str_index_list])
                    entity_list.append(entity_str)
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                start_end_tuple_list.append((i, i))
                entity_list.append(entity_str)
        return entity_list, start_end_tuple_list


class fastPredictMRC:
    def __init__(self, model_path, config, model_type):
        self.model_path = model_path
        self.model_type = model_type
        self.data_loader = self.init_data_loader(config, model_type)
        # self.predict_fn = self.load_models()
        self.config = config

    def load_models(self, model_path):
        subdirs = [x for x in Path(model_path).iterdir()
                   if x.is_dir() and 'tmp' not in str(x)]
        latest = str(sorted(subdirs)[-1]) # take the latest, sorted by the suffix number:"1606125131"
        # output/model/wwm_lr_fold_0_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model/1606125131
        print(latest)
        predict_fn = predictor.from_saved_model(latest)
        
        return predict_fn

    def init_data_loader(self, config, model_type):
        vocab_file_path = os.path.join(config.get(
            "bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(event_config.get("slot_list_root_path"),
                                     event_config.get("bert_slot_complete_file_name_role"))
        schema_file = os.path.join(event_config.get(
                "data_dir"), event_config.get("event_schema"))
        query_map_file = os.path.join(event_config.get(
                "slot_list_root_path"), event_config.get("query_map_file"))
        data_loader = EventRolePrepareMRC(
                vocab_file_path, 512, slot_file, schema_file, query_map_file)

        return data_loader

    def parse_test_json(self, test_file):
        id_list = []
        text_list = []
        with codecs.open(test_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                d_json = json.loads(line)
                id_list.append(d_json.get("id"))
                text_list.append(d_json.get("text"))
        return id_list, text_list

    def predict_single_sample(self, predict_fn, text, query_len, token_type_ids):
        text_length = len(text)
        # print(text)

        predictions = predict_fn({'words': [text], 'text_length': [text_length], 'query_length': [query_len],
                                  'token_type_ids': [token_type_ids]})
        # predictions = predict_fn({'words': [text], 'text_length': [text_length],
        #                           'token_type_ids': [token_type_ids]})
        # print(predictions)
        # start_ids, end_ids,start_probs,end_probs = predictions.get("start_ids"), predictions.get("end_ids"),predictions.get("start_probs"), predictions.get("end_probs")
        # print("Debug Output!!!!!!!!!!!")
        # print("predictions")
        # print(predictions)
        
        # pred_ids, pred_probs = predictions.get("pred_ids"), predictions.get("pred_probs")
        # return start_ids[0], end_ids[0],start_probs[0], end_probs[0]
        
        # return pred_ids[0], pred_probs[0]
        # pred_probs = predictions.get("output")
        # return None, pred_probs[0]
        start_ids = predictions.get("start_ids")
        end_ids = predictions.get("end_ids")
        return start_ids, end_ids

    def extract_entity_from_start_end_ids(self, text, start_ids, end_ids, token_mapping):
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        # text_cur_index = 0
        for i, start_id in enumerate(start_ids):
            if start_id == 0:
                # text_cur_index += len(token_mapping[i])
                continue
            if end_ids[i] == 1:
                # start and end is the same
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                # entity_str = text[text_cur_index:text_cur_index+cur_entity_len]
                entity_list.append(entity_str)
                # text_cur_index += len(token_mapping[i])
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids):
                # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
                if start_ids[j] == 1:
                    break
                if end_ids[j] == 1:
                    entity_str_index_list = []
                    for index in range(i, j + 1):
                        entity_str_index_list.extend(token_mapping[index])

                    entity_str = "".join([text[char_index]
                                          for char_index in entity_str_index_list])
                    entity_list.append(entity_str)
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                entity_list.append(entity_str)
        return entity_list

def parse_event_schema(json_file):
    event_type_role_dict = {}
    with codecs.open(json_file, 'r', 'utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = line.strip("\n")
            data_json = json.loads(line)
            event_type = data_json.get("event_type")
            role_list = data_json.get("role_list")
            role_name_list = [ele.get("role") for ele in role_list]
            event_type_role_dict.update({event_type: role_name_list})
    return event_type_role_dict


def parse_schema_type(schema_file):
    schema_event_dict = {}
    with codecs.open(schema_file, 'r', 'utf-8') as fr:
        for line in fr:
            s_json = json.loads(line)
            role_list = s_json.get("role_list")
            schema_event_dict.update(
                {s_json.get("event_type"): [ele.get("role") for ele in role_list]})
    return schema_event_dict


def extract_entity_span_from_muliclass(text, pred_ids, token_mapping):
    buffer_list = []
    entity_list = []
    for index, label in enumerate(pred_ids):
        if label == 0:
            if buffer_list:
                entity_str_index_list = []
                for i in buffer_list:
                    entity_str_index_list.extend(token_mapping[i])
                entity_str = "".join([text[char_index]
                                      for char_index in entity_str_index_list])
                entity_list.append(entity_str)
                buffer_list.clear()
            continue
        elif label == 1:
            if buffer_list:
                entity_str_index_list = []
                for i in buffer_list:
                    entity_str_index_list.extend(token_mapping[i])
                entity_str = "".join([text[char_index]
                                      for char_index in entity_str_index_list])
                entity_list.append(entity_str)
                buffer_list.clear()
            buffer_list.append(index)
        else:
            if buffer_list:
                buffer_list.append(index)
    if buffer_list:
        entity_str_index_list = []
        for i in buffer_list:
            entity_str_index_list.extend(token_mapping[i])
        entity_str = "".join([text[char_index]
                              for char_index in entity_str_index_list])
        entity_list.append(entity_str)
    return entity_list


def parse_kfold(args):
    # 'data/test.json'
    test_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_test"))
    # 'output/model/index_fold_{}_roberta_large_traindev_desmodified_lowercase_event_type_class_bert_model/saved_mode'
    class_type_model_path = event_config.get(args.event_type_model_path)
    # 'data/event_schema.json'
    event_schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))

    event_schema_dict = parse_event_schema(event_schema_file)
    fp_type = fastPredictTypeClassification(class_type_model_path, event_config)
    id_list, text_list = fp_type.parse_test_json(test_file)

    
    # print("Debug Output!!!!!!!!!!!")
    # print("len ")
    # print(text_list[-1])
    # exit(1)
    
    kfold_type_result_list = []
    event_type_result_list = []

    ## 用k个fold 的模型 运行，这里使用了 六轮交叉验证的模型融合和我不同超参数模型的概率融合, 分析每个文本包含什么事件

    for k in range(1):
        predict_fn = fp_type.load_models_kfold(class_type_model_path.format(k))
        cur_fold_event_type_probs = fp_type.predict_for_all_prob(predict_fn, text_list)
        kfold_type_result_list.append(cur_fold_event_type_probs)
    

    for i in range(len(text_list)):
        cur_sample_event_type_buffer = [ele[i] for ele in kfold_type_result_list]
        cur_sample_event_type_prob = np.array(cur_sample_event_type_buffer).reshape((-1, 65))
        avg_result = np.mean(cur_sample_event_type_prob, axis=0)
        event_label_ids = np.argwhere(avg_result > 0.45)
        event_cur_type_strs = [fp_type.data_loader.id2labels_map.get(
            ele[0]) for ele in event_label_ids]
        event_type_result_list.append(event_cur_type_strs)
    
    # print("Debug Output!!!!!!!!!!!")
    # print("kfold_type_result_list!!!")
    # print(len(kfold_type_result_list)) ：k的数量
    # print(len(kfold_type_result_list[1])) ：1485
    # print(len(kfold_type_result_list[1][0])) ：65
    # exit(1)
    
    # event_type_result_list = fp_type.predict_for_all((text_list))
    # event_type_result_list = []
    # with codecs.open("new_final_event_type.txt", 'r', 'utf-8') as fr:
    #     for line in fr:
    #         line = line.strip("\n")
    #         event_list_cur = line.split(",")
    #         event_type_result_list.append(event_list_cur)
    
    role_model_path = event_config.get(args.model_role_pb_dir)
    # 'output/model/wwm_lr_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_mode

    role_model_path_use_best = "output/model/re_lr_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/checkpoint/export/best_exporter"
    fp_role_mrc = fastPredictMRC(role_model_path, event_config, "role")
    id_list, text_list = fp_role_mrc.parse_test_json(test_file)
    submit_result = []

    # index = 0
    kfold_result = []
    for k in range(1):
        # if k in [0,3,5]:
        # predict_fn 是一个 Predictor
        predict_fn = fp_role_mrc.load_models(role_model_path.format(k))

        # else:
        #     predict_fn = fp_role_mrc.load_models(role_model_path_use_best.format(k))
        cur_fold_probs_result = {}
        for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
            if event_type_res is None or len(event_type_res) == 0:
                # submit_result.append({"id": sample_id, "event_list": []})
                cur_fold_probs_result.update({sample_id: []})
                continue

            for cur_event_type in event_type_res:
                cur_event_type = cur_event_type.strip()
                if cur_event_type is None or cur_event_type == "":
                    continue
                corresponding_role_type_list = event_schema_dict.get(cur_event_type)
                event_type_probs_result = [] # 
                for cur_role_type in corresponding_role_type_list:
                    cur_query_word = fp_role_mrc.data_loader.gen_query_for_each_sample(
                        cur_event_type, cur_role_type)
                    token_ids, query_len, token_type_ids, token_mapping = fp_role_mrc.data_loader.trans_single_data_for_test(
                        text, cur_query_word, 512)

                    # problems happens here!!!!
                    # pred_ids, pred_probs = fp_role_mrc.predict_single_sample(predict_fn, token_ids, query_len,
                                                                            #  token_type_ids)
                    start_ids , end_ids = fp_role_mrc.predict_single_sample(predict_fn, token_ids, query_len,
                                                                             token_type_ids)    

                    event_type_probs_result.append((start_ids, end_ids)) # the probs to answear the role question
                cur_fold_probs_result.update({sample_id + "-" + cur_event_type: event_type_probs_result}) 
                # save { sample1-event_type1:[0.2,0.3,0.4], sample1-event_type2:[0.2,0.9,0.8],
                # sample2-event_type3:[0.2,0.3,0,4], sample2-event_type4:[0.5,0.2,0.8,0.9,0.95] }
                # the main key is the combination of text and event_type
        kfold_result.append(cur_fold_probs_result)

    for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
        event_list = []
        if event_type_res is None or len(event_type_res) == 0:
            submit_result.append({"id": sample_id, "event_list": []})
            continue
        for cur_event_type in event_type_res:
            cur_event_type = cur_event_type.strip()
            if cur_event_type is None or cur_event_type == "":
                continue
            corresponding_role_type_list = event_schema_dict.get(cur_event_type)
            find_key = sample_id + "-" + cur_event_type
            fold_probs_cur_sample = [ele.get(find_key) for ele in kfold_result]
            for index, cur_role_type in enumerate(corresponding_role_type_list):
                ## multi-labels方式
                # cur_query_word = fp_role_mrc.data_loader.gen_query_for_each_sample(
                #     cur_event_type, cur_role_type)
                # token_ids, query_len, token_type_ids, token_mapping = fp_role_mrc.data_loader.trans_single_data_for_test(
                #     text, cur_query_word, 512)
                # cur_role_fold_probs = [probs[index] for probs in fold_probs_cur_sample]
                # # cur_role_fold_probs_array = np.vstack(cur_role_fold_probs)
                # token_len = len(token_ids)
                # cur_role_fold_probs_array = np.array(cur_role_fold_probs).reshape((1, token_len, 3))
                # avg_result = np.mean(cur_role_fold_probs_array, axis=0)
                # pred_ids = np.argmax(avg_result, axis=-1)
                # token_mapping = token_mapping[1:-1]
                # pred_ids = pred_ids[query_len:-1]
                # entity_list = extract_entity_span_from_muliclass(text, pred_ids, token_mapping)
                # for entity in entity_list:
                #     event_list.append({"event_type": cur_event_type, "arguments": [
                #         {"role": cur_role_type, "argument": entity}]})
                
                # start-labels和end-labels方式
                cur_query_word = fp_role_mrc.data_loader.gen_query_for_each_sample(
                    cur_event_type, cur_role_type)
                token_ids, query_len, token_type_ids, token_mapping = fp_role_mrc.data_loader.trans_single_data_for_test(
                    text, cur_query_word, 512)
                cur_role_fold_probs = [probs[index] for probs in fold_probs_cur_sample]
                # cur_role_fold_probs_array = np.vstack(cur_role_fold_probs)
                token_len = len(token_ids)
                cur_role_fold_probs_array = np.array(cur_role_fold_probs).reshape((1, token_len, 3))
                avg_result = np.mean(cur_role_fold_probs_array, axis=0)
                pred_ids = np.argmax(avg_result, axis=-1)
                token_mapping = token_mapping[1:-1]
                pred_ids = pred_ids[query_len:-1]
                entity_list = extract_entity_span_from_muliclass(text, pred_ids, token_mapping)
                for entity in entity_list:
                    event_list.append({"event_type": cur_event_type, "arguments": [
                        {"role": cur_role_type, "argument": entity}]})
        submit_result.append({"id": sample_id, "event_list": event_list})

    with codecs.open(args.submit_result, 'w', 'utf-8') as fw:
        for dict_result in submit_result:
            write_str = json.dumps(dict_result, ensure_ascii=False)
            fw.write(write_str)
            fw.write("\n")


def parse_kfold_verfify(args):
    # test_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_test"))
    test_file = "data/test1.json"
    class_type_model_path = event_config.get(args.event_type_model_path)
    event_schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))
    event_schema_dict = parse_event_schema(event_schema_file)
    fp_type = fastPredictTypeClassification(class_type_model_path, event_config)
    id_list, text_list = fp_type.parse_test_json(test_file)
    kfold_type_result_list = []
    event_type_result_list = []
    for k in range(6):
        predict_fn = fp_type.load_models_kfold(class_type_model_path.format(k))
        cur_fold_event_type_probs = fp_type.predict_for_all_prob(predict_fn,text_list)
        kfold_type_result_list.append(cur_fold_event_type_probs)

    for i in range(len(text_list)):
        cur_sample_event_type_buffer = [ele[i] for ele in kfold_type_result_list]
        cur_sample_event_type_prob = np.array(cur_sample_event_type_buffer).reshape((-1,65))
        avg_result = np.mean(cur_sample_event_type_prob,axis=0)
        event_label_ids = np.argwhere(avg_result > 0.5)
        event_cur_type_strs = [fp_type.data_loader.id2labels_map.get(
                ele[0]) for ele in event_label_ids]
        event_type_result_list.append(event_cur_type_strs)

    # with codecs.open("test2_kfold_new_final_event_type.txt", 'w', 'utf-8') as fw:
    #     for event_type_result in event_type_result_list:
    #         write_line = ",".join(event_type_result)
    #         fw.write(write_line)
    #         fw.write("\n")

    # event_type_result_list = []
    # with codecs.open("test2_kfold_new_final_event_type.txt", 'r', 'utf-8') as fr:
    #     for line in fr:
    #         line = line.strip("\n")
    #         event_list_cur = line.split(",")
    #         event_type_result_list.append(event_list_cur)
    # cls_model_path = event_config.get(args.event_cls_model_path)
    cls_model_path = "output/model/verify_cls_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    cls_model_path_new = "output/model/final_verify_cls_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    # verify_av_model_path_old = event_config.get(args.event_verfifyav_model_path)
    verify_av_model_path_old = "output/model/verify_avmrc_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    verify_av_model_path_new = "output/model/final_verify_avmrc_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    fp_cls_old = fastPredictCls(cls_model_path, event_config, "data/slot_pattern/slot_descrip_old")
    fp_cls_new = fastPredictCls(cls_model_path, event_config, "data/slot_pattern/slot_descrip")
    kfold_cls_result = []
    kfold_start_result = []
    kfold_end_result = []
    kfold_hasa_result = []
    for k in range(1):

        # predict_fn = fp_cls_old.load_models_kfold(cls_model_path.format(k))
        predict_fn_cls_new = fp_cls_new.load_models_kfold(cls_model_path_new.format(k))
        predict_fn_av = fp_cls_new.load_models_kfold(verify_av_model_path_new.format(k))
        # predict_fn_av_old = fp_cls_old.load_models_kfold(verify_av_model_path_old.format(k))
        cur_fold_cls_probs_result = {}
        cur_fold_av_start_probs_result = {}
        cur_fold_av_end_probs_result = {}
        cur_fold_av_has_answer_probs_result = {}
        for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):

            if event_type_res is None or len(event_type_res) == 0:
                # submit_result.append({"id": sample_id, "event_list": []})
                cur_fold_cls_probs_result.update({sample_id: []})
                continue
            for cur_event_type in event_type_res:
                cur_event_type = cur_event_type.strip()
                if cur_event_type is None or cur_event_type == "":
                    continue
                corresponding_role_type_list = event_schema_dict.get(cur_event_type)
                cur_event_type_cls_probs_result = []
                cur_event_av_start_probs_result = []
                cur_event_av_end_probs_result = []
                cur_event_av_hasanswer_probs_result = []
                for cur_role_type in corresponding_role_type_list:
                    # cur_query_word_old = fp_cls_old.data_loader.gen_query_for_each_sample(
                        # cur_event_type, cur_role_type)
                    # token_ids, query_len, token_type_ids, token_mapping = fp_cls_old.data_loader.trans_single_data_for_test(
                        # text, cur_query_word_old, 512)
                    # label_prob = fp_cls_old.predict_single_sample_prob(predict_fn, token_ids, query_len, token_type_ids)

                    # start_ids, end_ids, start_probs, end_probs, has_answer_probs = fp_cls_old.predict_single_sample_av_prob(
                        # predict_fn_av_old, token_ids, query_len, token_type_ids)
                    # cur_event_av_start_probs_result.append(start_probs)
                    # cur_event_av_end_probs_result.append(end_probs)
                    # new
                    has_answer_probs = None
                    label_prob = None
                    start_probs = None
                    end_probs = None
                    cur_query_word_new = fp_cls_new.data_loader.gen_query_for_each_sample(
                        cur_event_type, cur_role_type)
                    token_ids_new, query_len_new, token_type_ids_new, token_mapping_new = fp_cls_new.data_loader.trans_single_data_for_test(
                        text, cur_query_word_new, 512)
                    label_prob_new = fp_cls_new.predict_single_sample_prob(predict_fn_cls_new, token_ids_new,
                                                                           query_len_new, token_type_ids_new)
                    start_ids_new, end_ids_new, start_probs_new, end_probs_new, has_answer_probs_new = fp_cls_old.predict_single_sample_av_prob(
                        predict_fn_av, token_ids_new, query_len_new, token_type_ids_new)
                    cur_event_av_hasanswer_probs_result.append((has_answer_probs, has_answer_probs_new))
                    cur_event_type_cls_probs_result.append((label_prob, label_prob_new))
                    cur_event_av_start_probs_result.append((start_probs, start_probs_new))
                    cur_event_av_end_probs_result.append((end_probs, end_probs_new))
                cur_fold_cls_probs_result.update({sample_id + "-" + cur_event_type: cur_event_type_cls_probs_result})
                cur_fold_av_start_probs_result.update(
                    {sample_id + "-" + cur_event_type: cur_event_av_start_probs_result})
                cur_fold_av_end_probs_result.update({sample_id + "-" + cur_event_type: cur_event_av_end_probs_result})
                cur_fold_av_has_answer_probs_result.update(
                    {sample_id + "-" + cur_event_type: cur_event_av_hasanswer_probs_result})
        kfold_cls_result.append(cur_fold_cls_probs_result)
        kfold_start_result.append(cur_fold_av_start_probs_result)
        kfold_end_result.append(cur_fold_av_end_probs_result)
        kfold_hasa_result.append(cur_fold_av_has_answer_probs_result)

    submit_result = []
    for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
        event_list = []
        if event_type_res is None or len(event_type_res) == 0:
            submit_result.append({"id": sample_id, "event_list": []})
            continue
        for cur_event_type in event_type_res:
            cur_event_type = cur_event_type.strip()
            if cur_event_type is None or cur_event_type == "":
                continue
            corresponding_role_type_list = event_schema_dict.get(cur_event_type)
            find_key = sample_id + "-" + cur_event_type
            fold_cls_probs_cur_sample = [ele.get(find_key) for ele in kfold_cls_result]
            fold_start_probs_cur_sample = [ele.get(find_key) for ele in kfold_start_result]
            fold_end_probs_cur_sample = [ele.get(find_key) for ele in kfold_end_result]
            fold_has_probs_cur_sample = [ele.get(find_key) for ele in kfold_hasa_result]
            for index, cur_role_type in enumerate(corresponding_role_type_list):
                cur_cls_fold_probs = [probs[index] for probs in fold_cls_probs_cur_sample]
                cur_cls_fold_probs_old = []
                cur_cls_fold_probs_new = []
                cur_hasa_fold_probs = [probs[index] for probs in fold_has_probs_cur_sample]
                cur_hasa_fold_probs_old = []
                cur_hasa_fold_probs_new = []
                for k in range(len(cur_cls_fold_probs)):
                    # cur_cls_fold_probs_old.append(cur_cls_fold_probs[k][0])
                    cur_cls_fold_probs_new.append(cur_cls_fold_probs[k][1])
                    # cur_hasa_fold_probs_old.append(cur_hasa_fold_probs[k][0])
                    cur_hasa_fold_probs_new.append(cur_hasa_fold_probs[k][1])

                # cur_cls_fold_probs_old = np.array(cur_cls_fold_probs_old).reshape((6, 1))
                # cls_avg_result_old = np.mean(cur_cls_fold_probs_old, axis=0)

                cur_cls_fold_probs_new = np.array(cur_cls_fold_probs_new).reshape((-1, 1))
                cls_avg_result_new = np.mean(cur_cls_fold_probs_new, axis=0)

                # cur_hasa_fold_probs_old = np.array(cur_hasa_fold_probs_old).reshape((6, 1))
                # has_avg_result_old = np.mean(cur_hasa_fold_probs_old, axis=0)

                cur_hasa_fold_probs_new = np.array(cur_hasa_fold_probs_new).reshape((-1, 1))
                has_avg_result_new = np.mean(cur_hasa_fold_probs_new, axis=0)

                # cur_hasa_fold_probs = np.array(cur_hasa_fold_probs).reshape((6,1))
                # has_avg_result = np.mean(cur_hasa_fold_probs,axis=0)
                # final_probs_hasa = 0.5 * (cls_avg_result_old + cls_avg_result_new) / 2 + 0.5 * (
                #             has_avg_result_old + has_avg_result_new) / 2
                final_probs_hasa = 0.5 * (cls_avg_result_new) + 0.5 * (has_avg_result_new)

                if final_probs_hasa > 0.4:
                    cur_query_word = fp_cls_new.data_loader.gen_query_for_each_sample(
                        cur_event_type, cur_role_type)
                    token_ids, query_len, token_type_ids, token_mapping = fp_cls_new.data_loader.trans_single_data_for_test(
                        text, cur_query_word, 512)

                    # cur_query_word_old = fp_cls_old.data_loader.gen_query_for_each_sample(
                    #     cur_event_type, cur_role_type)
                    # token_ids_old, query_len_old, token_type_ids_old, token_mapping_old = fp_cls_old.data_loader.trans_single_data_for_test(
                    #     text, cur_query_word_old, 512)

                    token_len = len(token_ids)
                    # token_len_old = len(token_ids_old)
                    cur_start_fold_probs = [probs[index] for probs in fold_start_probs_cur_sample]
                    cur_end_fold_probs = [probs[index] for probs in fold_end_probs_cur_sample]
                    cur_start_fold_probs_old = []
                    cur_start_fold_probs_new = []
                    cur_end_fold_probs_old = []
                    cur_end_fold_probs_new = []

                    for k in range(len(cur_start_fold_probs)):
                        # cur_start_fold_probs_old.append(cur_start_fold_probs[k][0])
                        cur_start_fold_probs_new.append(cur_start_fold_probs[k][1])
                        # cur_end_fold_probs_old.append(cur_end_fold_probs[k][0])
                        cur_end_fold_probs_new.append(cur_end_fold_probs[k][1])
                    # cur_start_fold_probs_old = [probs[index] for probs in fold_start_probs_cur_sample]
                    # cur_end_fold_probs_old = [probs[index] for probs in fold_end_probs_cur_sample]
                    # cur_start_fold_probs_old = np.array(cur_start_fold_probs_old).reshape((6, token_len_old, 2))
                    # cur_end_fold_probs_old = np.array(cur_end_fold_probs_old).reshape((6, token_len_old, 2))
                    # start_avg_result_old = np.mean(cur_start_fold_probs_old, axis=0)
                    # end_avg_result_old = np.mean(cur_end_fold_probs_old, axis=0)

                    # pos_start_probs_old = start_avg_result_old[:, 1]
                    # pos_end_probs_old = end_avg_result_old[:, 1]
                    # text_start_probs_old = pos_start_probs_old[query_len_old:-1]
                    # text_end_probs_old = pos_end_probs_old[query_len_old:-1]

                    cur_start_fold_probs_new = np.array(cur_start_fold_probs_new).reshape((-1, token_len, 2))
                    cur_end_fold_probs_new = np.array(cur_end_fold_probs_new).reshape((-1, token_len, 2))
                    start_avg_result_new = np.mean(cur_start_fold_probs_new, axis=0)
                    end_avg_result_new = np.mean(cur_end_fold_probs_new, axis=0)

                    pos_start_probs_new = start_avg_result_new[:, 1]
                    pos_end_probs_new = end_avg_result_new[:, 1]
                    text_start_probs_new = pos_start_probs_new[query_len:-1]
                    text_end_probs_new = pos_end_probs_new[query_len:-1]

                    # pos_start_probs = (text_start_probs_old + text_start_probs_new) / 2
                    # pos_end_probs = (text_end_probs_old + text_end_probs_new) / 2
                    pos_start_probs = (text_start_probs_new) 
                    pos_end_probs = (text_end_probs_new)

                    start_ids = (pos_start_probs > 0.4).astype(int)
                    # end_ids = np.argmax(end_avg_result,axis=-1)
                    end_ids = (pos_end_probs > 0.4).astype(int)
                    token_mapping = token_mapping[1:-1]
                    # start_ids = start_ids[query_len:-1]

                    # end_ids = end_ids[query_len:-1]

                    entity_list, span_start_end_tuple_list = fp_cls_new.extract_entity_from_start_end_ids(
                        text=text, start_ids=start_ids, end_ids=end_ids, token_mapping=token_mapping)
                    # if len(entity_list) == 0:
                    #     score_has_answer = 0.0
                    # else:
                    #     span_score = [text_start_probs[ele[0]]+text_end_probs[ele[1]] for ele in span_start_end_tuple_list]
                    #     score_has_answer = max(span_score)
                    # score_no_answer = 0.5*(max(pos_start_probs[0:query_len])+max(pos_end_probs[0:query_len]))+0.5*final_probs_hasa
                    # diff_score = score_has_answer - score_no_answer
                    for entity in entity_list:
                        if len(entity) > 1:
                            event_list.append({"event_type": cur_event_type, "arguments": [
                                {"role": cur_role_type, "argument": entity}]})
        submit_result.append({"id": sample_id, "event_list": event_list})
    # for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
    #     event_list = []
    #     if event_type_res is None or len(event_type_res) == 0:
    #             submit_result.append({"id": sample_id, "event_list": []})
    #             continue
    #     for cur_event_type in event_type_res:
    #             cur_event_type = cur_event_type.strip()
    #             if cur_event_type is None or cur_event_type == "":
    #                 continue
    #             corresponding_role_type_list = event_schema_dict.get(cur_event_type)
    #             find_key = sample_id + "-" + cur_event_type
    #             fold_probs_cur_sample = [ele.get(find_key) for ele in kfold_result]
    #             for index,cur_role_type in enumerate(corresponding_role_type_list):
    #                 cur_query_word = fp_role_mrc.data_loader.gen_query_for_each_sample(
    #                     cur_event_type, cur_role_type)
    #                 token_ids, query_len, token_type_ids, token_mapping = fp_role_mrc.data_loader.trans_single_data_for_test(
    #                     text, cur_query_word, 512)
    #                 cur_role_fold_probs = [probs[index] for probs in fold_probs_cur_sample]
    #                 # cur_role_fold_probs_array = np.vstack(cur_role_fold_probs)
    #                 token_len = len(token_ids)
    #                 cur_role_fold_probs_array = np.array(cur_role_fold_probs).reshape((1,token_len,3))
    #                 avg_result = np.mean(cur_role_fold_probs_array,axis=0)
    #                 pred_ids = np.argmax(avg_result,axis=-1)
    #                 token_mapping = token_mapping[1:-1]
    #                 pred_ids = pred_ids[query_len:-1]
    #                 entity_list = extract_entity_span_from_muliclass(text,pred_ids,token_mapping)
    #                 for entity in entity_list:
    #                     event_list.append({"event_type": cur_event_type, "arguments": [
    #                                           {"role": cur_role_type, "argument": entity}]})
    #     submit_result.append({"id": sample_id, "event_list": event_list})

    # for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
    #     # if sample_id == "66de2f44ca8839ddcb0708096864df8b":
    #     #     print(text)
    #     if event_type_res is None or len(event_type_res) == 0:
    #         submit_result.append({"id": sample_id, "event_list": []})
    #         continue
    #     event_list = []
    #     # print(event_type_res)
    #     # {"event_type": "司法行为-开庭", "arguments": [{"role": "时间", "argument": "4月29日上午"}
    #     for cur_event_type in event_type_res:
    #         cur_event_type = cur_event_type.strip()
    #         if cur_event_type is None or cur_event_type == "":
    #             continue
    #         corresponding_role_type_list = event_schema_dict.get(
    #             cur_event_type)
    #         for cur_role_type in corresponding_role_type_list:
    #             if True:
    #                 cur_query_word = fp_role_mrc.data_loader.gen_query_for_each_sample(
    #                     cur_event_type, cur_role_type)
    #                 token_ids, query_len, token_type_ids, token_mapping = fp_role_mrc.data_loader.trans_single_data_for_test(
    #                     text, cur_query_word, 512)
    #                 start_ids, end_ids,start_probs,end_probs = fp_role_mrc.predict_single_sample(
    #                     token_ids, query_len, token_type_ids)
    #                 # print(start_probs.shape)
    #                 pos_start_probs = start_probs[:,1]
    #                 # pos_end_probs = end_probs[:,1]
    #                 start_ids = (pos_start_probs > 0.4).astype(int)
    #                 # end_ids = (pos_end_probs > 0.4).astype(int)
    #                 # end_ids = (pos_end_probs > 0.4).astype(int)
    #                 # 先松后紧
    #                 if sum(start_ids) == 0:
    #                     continue
    #                 # if sum(start_ids) > 1:
    #                 #     print(text)
    #                 token_mapping = token_mapping[1:-1]
    #                 # a = start_ids[query_len-1:]
    #                 start_ids = start_ids[query_len:-1]
    #                 end_ids = end_ids[query_len:-1]
    #                 entity_list = fp_role_mrc.extract_entity_from_start_end_ids(
    #                     text=text, start_ids=start_ids, end_ids=end_ids, token_mapping=token_mapping)
    #                 for entity in entity_list:
    #                     if len(entity) > 1:
    #                         event_list.append({"event_type": cur_event_type, "arguments": [
    #                                           {"role": cur_role_type, "argument": entity}]})
    #     submit_result.append({"id": sample_id, "event_list": event_list})

    with codecs.open(args.submit_result, 'w', 'utf-8') as fw:
        for dict_result in submit_result:
            write_str = json.dumps(dict_result, ensure_ascii=False)
            fw.write(write_str)
            fw.write("\n")


if __name__ == '__main__':
    print(os.listdir("data/slot_pattern/"))
    parser = ArgumentParser()
    parser.add_argument("--mode", default="verify", type=str)
    parser.add_argument("--model_trigger_pb_dir",
                        default='bert_model_pb', type=str)
    parser.add_argument("--model_role_pb_dir",
                        default='role_bert_model_pb', type=str)
    parser.add_argument("--trigger_predict_res",
                        default="trigger_result.json", type=str)
    parser.add_argument("--submit_result",
                        default="test2allmerge_modifeddes_prob4null_0404threold_8epoch_type05_verify_kfold_notav_modifyneg_dropout15moretype_roberta_large.json",
                        type=str)
    parser.add_argument("--multi_task_model_pb_dir",
                        default="multi_task_bert_model_pb", type=str)
    parser.add_argument("--event_type_model_path",
                        default="type_class_bert_model_pb", type=str)
    parser.add_argument("--event_cls_model_path",
                        default="role_verify_cls_bert_model_pb", type=str)
    parser.add_argument("--event_verfifyav_model_path",
                        default="role_verify_avmrc_bert_model_pb", type=str)

    # parser.add_argument("--model_pb_dir", default='base_pb_model_dir', type=str)
    args = parser.parse_args()
    # print(args.label_less)
    # parse_main(args)
    if(args.mode=="verify"):
        parse_kfold_verfify(args)
    elif(args.mode=="test1"):
        parse_kfold(args)

    # parse_multitask(args)
