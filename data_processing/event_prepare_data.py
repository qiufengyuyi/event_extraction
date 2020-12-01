import codecs
import json
import tensorflow as tf
import random
import numpy as np
import re
import os
# deprecated
#from data_processing.tokenize import EventTokenizer
from bert4keras.tokenizers import Tokenizer
from configs.event_config import event_config

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class EventRolePrepareMRC:
    def __init__(self, vocab_file, max_seq_length, labels_file, schema_file, query_des_file):
        self.max_seq_length = max_seq_length
        self.schema_dict = self.parse_schema_type(schema_file)
        self.labels_map, self.id2labels_map = self.read_slot(labels_file)
        self.query_map = self.query_str_gen(query_des_file)
        self.labels_map_len = len(self.labels_map)
        self.tokenizer = Tokenizer(vocab_file, do_lower_case=True)

    def query_str_gen(self, query_des_file):
        query_map = {}
        with codecs.open(query_des_file, 'r', 'utf-8') as fr:
            for index, line in enumerate(fr):
                line = line.strip("\n")
                line = line.strip("\r")
                query_map.update({index: line})
        return query_map

    def read_slot(self, slot_file):
        labels_map = {}
        id2labels_map = {}
        with codecs.open(slot_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                arr = line.split("\t")
                labels_map[arr[0]] = int(arr[1])
                id2labels_map[int(arr[1])] = arr[0]
        return labels_map, id2labels_map

    def _read_json_file(self, train_input_file, eval_input_file, is_train):
        """_read_json_file"""
        train_input_data = []
        with codecs.open(train_input_file, "r", encoding='utf-8') as f:
            for line in f:
                d_json = json.loads(line.strip())
                train_input_data.append(d_json)
        if is_train:
            with codecs.open(eval_input_file, "r", encoding='utf-8') as f:
                for line in f:
                    d_json = json.loads(line.strip())
                    train_input_data.append(d_json)
        # train_data_list,train_label_start_list,train_label_end_list,train_query_len_list,train_token_type_id_list,dev_data_list,dev_label_start_list,dev_label_end_list,dev_query_len_list,dev_token_type_id_list = self.parse_data_from_json(train_input_data,is_train,False)
        # return train_data_list,train_label_start_list,train_label_end_list,train_query_len_list,train_token_type_id_list,dev_data_list,dev_label_start_list,dev_label_end_list,dev_query_len_list,dev_token_type_id_list
        data_list, label_start_list, label_end_list, multi_label_list, query_len_list, token_type_id_list, has_answer_list, data_list_neg, label_start_list_neg, label_end_list_neg, multi_label_list_neg, query_len_list_neg, token_type_id_list_neg, has_answer_list_neg = self.parse_data_from_json(
            train_input_data, is_train)
        return data_list, label_start_list, label_end_list, multi_label_list, query_len_list, token_type_id_list, has_answer_list, data_list_neg, label_start_list_neg, label_end_list_neg, multi_label_list_neg, query_len_list_neg, token_type_id_list_neg, has_answer_list_neg

    def k_fold_split_data(self, train_input_file, eval_input_file, re_train_file=None, re_dev_file=None, is_train=True,
                          fold_num=6):
        # re_data_list,re_multi_label,_,_,re_query_len_list,re_token_type_id_list = self._read_json_relation_file(re_train_file,re_dev_file)
        data_list, label_start_list, label_end_list, multi_label_list, query_len_list, token_type_id_list, has_answer_list, data_list_neg, label_start_list_neg, label_end_list_neg, multi_label_list_neg, query_len_list_neg, token_type_id_list_neg, has_answer_list_neg = self._read_json_file(
            train_input_file, eval_input_file, is_train)
        random.seed(2)
        # re_index_list = [i for i in range(len(re_data_list))]
        # random.shuffle(re_index_list)
        # re_data_list = [re_data_list[index] for index in re_index_list]
        # re_multi_label = [re_multi_label[index] for index in re_index_list]
        # re_query_len_list = [re_query_len_list[index] for index in re_index_list]
        # re_token_type_id_list = [re_token_type_id_list[index] for index in re_index_list]
        pos_num, neg_num = len(data_list), len(data_list_neg)
        dev_num, dev_neg_num = int(pos_num / fold_num), int(neg_num / fold_num)

        pos_index_list = [i for i in range(pos_num)]
        neg_index_list = [i for i in range(neg_num)]

        random.shuffle(pos_index_list)
        random.shuffle(neg_index_list)
        data_list = [data_list[index] for index in pos_index_list]
        label_start_list = [label_start_list[index] for index in pos_index_list]
        label_end_list = [label_end_list[index] for index in pos_index_list]
        multi_label_list = [multi_label_list[index] for index in pos_index_list]
        query_len_list = [query_len_list[index] for index in pos_index_list]
        token_type_id_list = [token_type_id_list[index] for index in pos_index_list]
        data_list_neg = [data_list_neg[index] for index in neg_index_list]
        multi_label_list_neg = [multi_label_list_neg[index] for index in neg_index_list]
        label_start_list_neg = [label_start_list_neg[index] for index in neg_index_list]
        label_end_list_neg = [label_end_list_neg[index] for index in neg_index_list]
        query_len_list_neg = [query_len_list_neg[index] for index in neg_index_list]
        token_type_id_list_neg = [token_type_id_list_neg[index] for index in neg_index_list]
        for fold_index in range(fold_num):

            if not os.path.exists("data/verify_neg_fold_data_{}".format(fold_index)):
                os.mkdir("data/verify_neg_fold_data_{}".format(fold_index))
            pos_fold_start = dev_num * fold_index
            neg_fold_start = dev_neg_num * fold_index
            if fold_index == 0:
                neg_fold_end = neg_fold_start + dev_neg_num
                pos_fold_end = pos_fold_start + dev_num
                cur_data_list_dev = data_list[pos_fold_start:pos_fold_end] + data_list_neg[neg_fold_start:neg_fold_end]
                cur_data_list_train = data_list[pos_fold_end:] + data_list_neg[neg_fold_end:]
                cur_label_start_list_dev = label_start_list[pos_fold_start:pos_fold_end] + label_start_list_neg[
                                                                                           neg_fold_start:neg_fold_end]
                cur_label_start_list_train = label_start_list[pos_fold_end:] + label_start_list_neg[neg_fold_end:]
                cur_label_end_list_dev = label_end_list[pos_fold_start:pos_fold_end] + label_end_list_neg[
                                                                                       neg_fold_start:neg_fold_end]
                cur_label_end_list_train = label_end_list[pos_fold_end:] + label_end_list_neg[neg_fold_end:]
                cur_has_answer_dev = has_answer_list[pos_fold_start:pos_fold_end] + has_answer_list_neg[
                                                                                    neg_fold_start:neg_fold_end]
                cur_has_answer_train = has_answer_list[pos_fold_end:] + has_answer_list_neg[neg_fold_end:]
                cur_multi_label_list_dev = multi_label_list[pos_fold_start:pos_fold_end] + multi_label_list_neg[
                                                                                           neg_fold_start:neg_fold_end]
                cur_multi_label_list_train = multi_label_list[pos_fold_end:] + multi_label_list_neg[neg_fold_end:]
                cur_query_len_list_dev = query_len_list[pos_fold_start:pos_fold_end] + query_len_list_neg[
                                                                                       neg_fold_start:neg_fold_end]
                cur_query_len_list_train = query_len_list[pos_fold_end:] + query_len_list_neg[neg_fold_end:]
                cur_token_type_id_list_dev = token_type_id_list[pos_fold_start:pos_fold_end] + token_type_id_list_neg[
                                                                                               neg_fold_start:neg_fold_end]
                cur_token_type_id_list_train = token_type_id_list[pos_fold_end:] + token_type_id_list_neg[neg_fold_end:]
            elif fold_index == fold_num - 1:
                cur_data_list_dev = data_list[pos_fold_start:] + data_list_neg[neg_fold_start:]
                cur_data_list_train = data_list[0:pos_fold_start] + data_list_neg[0:neg_fold_start]
                cur_label_start_list_dev = label_start_list[pos_fold_start:] + label_start_list_neg[neg_fold_start:]
                cur_label_start_list_train = label_start_list[0:pos_fold_start] + label_start_list_neg[0:neg_fold_start]
                cur_label_end_list_dev = label_end_list[pos_fold_start:] + label_end_list_neg[neg_fold_start:]
                cur_label_end_list_train = label_end_list[0:pos_fold_start] + label_end_list_neg[0:neg_fold_start]
                cur_has_answer_dev = has_answer_list[pos_fold_start:] + has_answer_list_neg[neg_fold_start:]
                cur_has_answer_train = has_answer_list[0:pos_fold_start] + has_answer_list_neg[0:neg_fold_start]
                cur_multi_label_list_dev = multi_label_list[pos_fold_start:] + multi_label_list_neg[neg_fold_start:]
                cur_multi_label_list_train = multi_label_list[0:pos_fold_start] + multi_label_list_neg[0:neg_fold_start]
                cur_query_len_list_dev = query_len_list[pos_fold_start:] + query_len_list_neg[neg_fold_start:]
                cur_query_len_list_train = query_len_list[0:pos_fold_start] + query_len_list_neg[0:neg_fold_start]
                cur_token_type_id_list_dev = token_type_id_list[pos_fold_start:] + token_type_id_list_neg[
                                                                                   neg_fold_start:]
                cur_token_type_id_list_train = token_type_id_list[0:pos_fold_start] + token_type_id_list_neg[
                                                                                      0:neg_fold_start]
            else:
                neg_fold_end = neg_fold_start + dev_neg_num
                pos_fold_end = pos_fold_start + dev_num
                cur_data_list_dev = data_list[pos_fold_start:pos_fold_end] + data_list_neg[neg_fold_start:neg_fold_end]
                cur_data_list_train = data_list[0:pos_fold_start] + data_list[pos_fold_end:] + data_list_neg[
                                                                                               0:neg_fold_start] + data_list_neg[
                                                                                                                   neg_fold_end:]
                cur_label_start_list_dev = label_start_list[pos_fold_start:pos_fold_end] + label_start_list_neg[
                                                                                           neg_fold_start:neg_fold_end]
                cur_label_start_list_train = label_start_list[0:pos_fold_start] + label_start_list[
                                                                                  pos_fold_end:] + label_start_list_neg[
                                                                                                   0:neg_fold_start] + label_start_list_neg[
                                                                                                                       neg_fold_end:]
                cur_label_end_list_dev = label_end_list[pos_fold_start:pos_fold_end] + label_end_list_neg[
                                                                                       neg_fold_start:neg_fold_end]
                cur_label_end_list_train = label_end_list[0:pos_fold_start] + label_end_list[
                                                                              pos_fold_end:] + label_end_list_neg[
                                                                                               0:neg_fold_start] + label_end_list_neg[
                                                                                                                   neg_fold_end:]
                cur_has_answer_dev = has_answer_list[pos_fold_start:pos_fold_end] + has_answer_list_neg[
                                                                                    neg_fold_start:neg_fold_end]
                cur_has_answer_train = has_answer_list[0:pos_fold_start] + has_answer_list[
                                                                           pos_fold_end:] + has_answer_list_neg[
                                                                                            0:neg_fold_start] + has_answer_list_neg[
                                                                                                                neg_fold_end:]
                cur_multi_label_list_dev = multi_label_list[pos_fold_start:pos_fold_end] + multi_label_list_neg[
                                                                                           neg_fold_start:neg_fold_end]
                cur_multi_label_list_train = multi_label_list[0:pos_fold_start] + multi_label_list[
                                                                                  pos_fold_end:] + multi_label_list_neg[
                                                                                                   0:neg_fold_start] + multi_label_list_neg[
                                                                                                                       neg_fold_end:]
                cur_query_len_list_dev = query_len_list[pos_fold_start:pos_fold_end] + query_len_list_neg[
                                                                                       neg_fold_start:neg_fold_end]
                cur_query_len_list_train = query_len_list[0:pos_fold_start] + query_len_list[
                                                                              pos_fold_end:] + query_len_list_neg[
                                                                                               0:neg_fold_start] + query_len_list_neg[
                                                                                                                   neg_fold_end:]
                cur_token_type_id_list_dev = token_type_id_list[pos_fold_start:pos_fold_end] + token_type_id_list_neg[
                                                                                               neg_fold_start:neg_fold_end]
                cur_token_type_id_list_train = token_type_id_list[0:pos_fold_start] + token_type_id_list[
                                                                                      pos_fold_end:] + token_type_id_list_neg[
                                                                                                       0:neg_fold_start] + token_type_id_list_neg[
                                                                                                                           neg_fold_end:]

            # cur_data_list_train += re_data_list[0:15000]
            # cur_multi_label_list_train += re_multi_label[0:15000]
            # cur_query_len_list_train += re_query_len_list[0:15000]
            # cur_token_type_id_list_train += re_token_type_id_list[0:15000]
            train_data_index = [i for i in range(len(cur_data_list_train))]
            random.shuffle(train_data_index)
            cur_data_list_train = [cur_data_list_train[index] for index in train_data_index]
            cur_label_start_list_train = [cur_label_start_list_train[index] for index in train_data_index]
            cur_label_end_list_train = [cur_label_end_list_train[index] for index in train_data_index]
            cur_has_answer_train = [cur_has_answer_train[index] for index in train_data_index]
            cur_multi_label_list_train = [cur_multi_label_list_train[index] for index in train_data_index]
            cur_query_len_list_train = [cur_query_len_list_train[index] for index in train_data_index]
            cur_token_type_id_list_train = [cur_token_type_id_list_train[index] for index in train_data_index]
            np.save("data/verify_neg_fold_data_{}/token_ids_train.npy".format(fold_index), cur_data_list_train)
            np.save("data/verify_neg_fold_data_{}/token_ids_dev.npy".format(fold_index), cur_data_list_dev)
            np.save("data/verify_neg_fold_data_{}/multi_labels_train.npy".format(fold_index),
                    cur_multi_label_list_train)
            np.save("data/verify_neg_fold_data_{}/multi_labels_dev.npy".format(fold_index), cur_multi_label_list_dev)
            np.save("data/verify_neg_fold_data_{}/query_lens_train.npy".format(fold_index), cur_query_len_list_train)
            np.save("data/verify_neg_fold_data_{}/query_lens_dev.npy".format(fold_index), cur_query_len_list_dev)
            np.save("data/verify_neg_fold_data_{}/token_type_ids_train.npy".format(fold_index),
                    cur_token_type_id_list_train)
            np.save("data/verify_neg_fold_data_{}/token_type_ids_dev.npy".format(fold_index),
                    cur_token_type_id_list_dev)
            np.save("data/verify_neg_fold_data_{}/labels_start_dev.npy".format(fold_index), cur_label_start_list_dev)
            np.save("data/verify_neg_fold_data_{}/labels_start_train.npy".format(fold_index),  # 多注意这个标签
                    cur_label_start_list_train)
            np.save("data/verify_neg_fold_data_{}/labels_end_dev.npy".format(fold_index), cur_label_end_list_dev)
            np.save("data/verify_neg_fold_data_{}/labels_end_train.npy".format(fold_index), cur_label_end_list_train) # 多注意这个标签
            np.save("data/verify_neg_fold_data_{}/has_answer_dev.npy".format(fold_index), cur_has_answer_dev)
            np.save("data/verify_neg_fold_data_{}/has_answer_train.npy".format(fold_index), cur_has_answer_train)


    def parse_schema_type(self, schema_file):
        schema_event_dict = {}
        with codecs.open(schema_file, 'r', 'utf-8') as fr:
            for line in fr:
                s_json = json.loads(line)
                role_list = s_json.get("role_list")
                schema_event_dict.update({s_json.get("event_type"): [ele.get("role") for ele in role_list]})
        return schema_event_dict

    def tranform_singlg_data_example(self, sent, roles_list):
        words = list(sent)
        sent_ori_2_new_index = {}
        new_words = []
        new_start, new_end = -1, -1
        new_roles_list = {}
        for role_type, role in roles_list.items():
            new_roles_list[role_type] = {
                "role_type": role_type,
                "start": -1,
                "end": -1
            }

        for i, w in enumerate(words):
            for role_type, role in roles_list.items():
                if i == role["start"]:
                    new_roles_list[role_type]["start"] = len(new_words)
                if i == role["end"]:
                    new_roles_list[role_type]["end"] = len(new_words)

            if len(w.strip()) == 0:
                sent_ori_2_new_index[i] = -1
                for role_type, role in roles_list.items():
                    if i == role["start"]:
                        new_roles_list[role_type]["start"] += 1
                    if i == role["end"]:
                        new_roles_list[role_type]["end"] -= 1
            else:
                sent_ori_2_new_index[i] = len(new_words)
                new_words.append(w)
        for role_type, role in new_roles_list.items():
            if role["start"] > -1:
                role["text"] = u"".join(
                    new_words[role["start"]:role["end"] + 1])
            if role["end"] == len(new_words):
                role["end"] = len(new_words) - 1

        return [words, new_words, sent_ori_2_new_index, new_roles_list]

    def gen_query_for_each_sample(self, event_type, role_type):
        complete_slot_str = event_type + "-" + role_type
        slot_id = self.labels_map.get(complete_slot_str)
        query_str = self.query_map.get(slot_id)
        event_type_str = event_type.split("-")[-1]
        if query_str.__contains__("？"):
            query_str_final = query_str
        if query_str == role_type:
            query_str_final = "找到{}事件中的{}".format(event_type_str, role_type)
        elif role_type == "时间":
            query_str_final = "找到{}{}".format(event_type_str, query_str)
        else:
            query_str_final = "找到{}事件中的{},包括{}".format(event_type_str, role_type, query_str)
        return query_str_final

    def parse_data_from_json(self, input_data, is_train, merge=True):
        data_list = []
        label_start_list = []
        label_end_list = []
        query_len_list = []
        token_type_id_list = []
        data_list_neg = []
        label_start_list_neg = []
        label_end_list_neg = []
        multi_label_list = []
        multi_label_list_neg = []
        query_len_list_neg = []
        token_type_id_list_neg = []
        has_answer_list = []
        has_answer_list_neg = []
        index_list = [i for i in range(len(input_data))]
        random.shuffle(index_list)
        event_type_record_list = []
        role_not_included_list = []
        in_role_dict = {}
        no_role_dict = {}
        no_role_list = []
        for index in index_list:
            data = input_data[index]
            sentence = data["text"]
            sentence_token_ids, sentence_token_type_ids = self.tokenizer.encode(sentence)
            if len(sentence_token_ids) != len(sentence_token_type_ids):
                print(sentence)
            sentence_token_type_ids = [ids + 1 for ids in sentence_token_type_ids] # 为拼接准备
            if len(sentence_token_ids) != len(sentence_token_type_ids):
                print(sentence)
            event_list = data["event_list"]

            event_record_list = [event_ele.get("event_type") for event_ele in event_list] # 没有用
            event_type_record_list.append(list(set(event_record_list))) # 没有用
            for event in event_list:
                dealt_role_list = []
                event_type = event["event_type"]
                event_type_words = [w for w in event_type]
                roles_list = {}
                role_type_dict = {}
                for index, role in enumerate(event["arguments"]):
                    role_type = role["role"]
                    if role_type in role_type_dict:
                        role_type_dict.get(role_type).append(index)
                    else:
                        role_type_dict.update({role_type: [index]})
                for role_type_key, argument_index_list in role_type_dict.items():

                    dealt_role_list.append(role_type_key)
                    query_word = self.gen_query_for_each_sample(event_type, role_type_key)
                    query_word_token_ids, query_word_token_type_ids = self.tokenizer.encode(query_word)
                    cur_start_labels = [0] * len(sentence_token_ids)
                    cur_end_labels = [0] * len(sentence_token_ids)
                    cur_BIO_labels = [0] * len(sentence_token_ids)
                    for argument_index in argument_index_list:

                        role_argument = event["arguments"][argument_index]
                        role_text = role_argument["argument"]
                        a_token_ids = self.tokenizer.encode(role_text)[0][1:-1]
                        start_index = search(a_token_ids, sentence_token_ids)
                        role_end = start_index + len(a_token_ids) - 1
                        # binary class
                        if start_index != -1:
                            cur_start_labels[start_index] = 1
                            cur_end_labels[role_end] = 1
                        # multi-class BIO
                        if start_index != -1:
                            # B
                            cur_BIO_labels[start_index] = 1
                            if role_end > start_index:
                                for i in range(start_index + 1, role_end + 1):
                                    cur_BIO_labels[i] = 2

                    cur_final_token_ids = query_word_token_ids + sentence_token_ids[1:]
                    cur_final_token_type_ids = query_word_token_type_ids + sentence_token_type_ids[1:]
                    cur_start_labels = [0] * len(query_word_token_ids) + cur_start_labels[1:]
                    cur_end_labels = [0] * len(query_word_token_ids) + cur_end_labels[1:]
                    cur_BIO_labels = [0] * len(query_word_token_ids) + cur_BIO_labels[1:]

                    data_list.append(cur_final_token_ids)
                    label_start_list.append(cur_start_labels)
                    label_end_list.append(cur_end_labels)
                    multi_label_list.append(cur_BIO_labels)
                    query_len_list.append(len(query_word_token_ids))
                    token_type_id_list.append(cur_final_token_type_ids)
                    has_answer_list.append(1)
                    if role_type_key in in_role_dict:
                        in_role_dict[role_type_key] += 1
                    else:
                        in_role_dict[role_type_key] = 1

                if is_train:
                    schema_role_list = self.schema_dict.get(event_type)
                    # cur_tmp_dict = {"text":sentence}
                    # tmp_list = []
                    for schema_role in schema_role_list:
                        if schema_role not in dealt_role_list:
                            # tmp_list.append(schema_role)
                            if True:
                                #     print(data)
                                query_word = self.gen_query_for_each_sample(event_type, schema_role)
                                if len(query_word) < 15 and not query_word.__contains__("？"):
                                    continue
                                query_word_token_ids, query_word_token_type_ids = self.tokenizer.encode(query_word)
                                cur_final_token_ids = query_word_token_ids + sentence_token_ids[1:]
                                cur_final_token_type_ids = query_word_token_type_ids + sentence_token_type_ids[1:]
                                cur_start_labels = [0] * len(cur_final_token_ids)
                                cur_end_labels = [0] * len(cur_final_token_ids)
                                cur_BIO_labels = [0] * len(cur_final_token_ids)
                                # cur_BIO_labels[-1] = 1
                                data_list_neg.append(cur_final_token_ids)
                                label_start_list_neg.append(cur_start_labels)
                                label_end_list_neg.append(cur_end_labels)
                                multi_label_list_neg.append(cur_BIO_labels)
                                query_len_list_neg.append(len(query_word_token_ids))
                                token_type_id_list_neg.append(cur_final_token_type_ids)
                                no_role_list.append(schema_role)
                                has_answer_list_neg.append(0)

        return data_list, label_start_list, label_end_list, multi_label_list, query_len_list, token_type_id_list, has_answer_list, data_list_neg, label_start_list_neg, label_end_list_neg, multi_label_list_neg, query_len_list_neg, token_type_id_list_neg, has_answer_list_neg


    def trans_single_data_for_test(self, text, query_word, max_seq_len):
        tokens = self.tokenizer.tokenize(text)
        # while len(tokens) > 510:
        #     tokens.pop(-2)
        query_token_ids, query_token_type_ids = self.tokenizer.encode(query_word)
        query_len = len(query_token_ids)

        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        while len(token_ids) - 1 > max_seq_len - query_len:
            token_ids.pop(-2)
        token_type_ids = [1] * len(token_ids)

        final_token_ids = query_token_ids + token_ids[1:]
        final_token_type_ids = query_token_type_ids + token_type_ids[1:]

        return final_token_ids, len(query_token_ids), final_token_type_ids, mapping


class EventTypeClassificationPrepare:
    def __init__(self, vocab_file, max_seq_length, all_event_types_file):
        self.max_seq_length = max_seq_length
        self.labels_map, self.id2labels_map = self.read_slot(all_event_types_file)
        self.labels_map_len = len(self.labels_map)
        self.tokenizer = Tokenizer(vocab_file, do_lower_case=True)

    def read_slot(self, slot_file):
        labels_map = {}
        id2labels_map = {}
        with codecs.open(slot_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                arr = line.split("\t")
                labels_map[arr[0]] = int(arr[1])
                id2labels_map[int(arr[1])] = arr[0]
        return labels_map, id2labels_map

    def truncate_seq_head_tail(self, tokens, head_len, tail_len, max_length):
        if len(tokens) <= max_length:
            return tokens
        else:
            head_tokens = tokens[0:head_len]
            tail_tokens = tokens[len(tokens) - tail_len:]
            return head_tokens + tail_tokens

    def k_fold_split_data(self, train_input_file, eval_input_file, is_train, fold_num=6):
        data_list, token_type_id_list, label_list, type_index_in_token_ids_list = self._read_json_file_kfold(
            train_input_file, eval_input_file, is_train)
        all_index_list = [i for i in range(len(data_list))]
        dev_num = int(len(data_list) / fold_num)
        random.seed(2)
        random.shuffle(all_index_list)
        data_list = [data_list[index] for index in all_index_list]
        label_list = [label_list[index] for index in all_index_list]
        token_type_id_list = [token_type_id_list[index] for index in all_index_list]
        type_index_in_token_ids_list = [type_index_in_token_ids_list[index] for index in all_index_list]
        for fold_index in range(fold_num):
            if not os.path.exists("data/index_type_fold_data_{}".format(fold_index)):
                os.mkdir("data/index_type_fold_data_{}".format(fold_index))
            fold_start = dev_num * fold_index
            if fold_index == 0:
                fold_end = fold_start + dev_num
                cur_data_list_dev = data_list[fold_start:fold_end]
                cur_data_list_train = data_list[fold_end:]
                cur_label_list_dev = label_list[fold_start:fold_end]
                cur_label_list_train = label_list[fold_end:]
                cur_token_type_id_list_dev = token_type_id_list[fold_start:fold_end]
                cur_token_type_id_list_train = token_type_id_list[fold_end:]
                cur_type_index_in_token_ids_list_dev = type_index_in_token_ids_list[fold_start:fold_end]
                cur_type_index_in_token_ids_list_train = type_index_in_token_ids_list[fold_end:]
            elif fold_index == fold_num - 1:
                cur_data_list_dev = data_list[fold_start:]
                cur_data_list_train = data_list[0:fold_start]
                cur_label_list_dev = label_list[fold_start:]
                cur_label_list_train = label_list[0:fold_start]
                cur_token_type_id_list_dev = token_type_id_list[fold_start:]
                cur_token_type_id_list_train = token_type_id_list[0:fold_start]
                cur_type_index_in_token_ids_list_dev = type_index_in_token_ids_list[fold_start:]
                cur_type_index_in_token_ids_list_train = type_index_in_token_ids_list[0:fold_start]
            else:
                fold_end = fold_start + dev_num
                cur_data_list_dev = data_list[fold_start:fold_end]
                cur_data_list_train = data_list[0:fold_start] + data_list[fold_end:]
                cur_label_list_dev = label_list[fold_start:fold_end]
                cur_label_list_train = label_list[0:fold_start] + label_list[fold_end:]
                cur_token_type_id_list_dev = token_type_id_list[fold_start:fold_end]
                cur_token_type_id_list_train = token_type_id_list[0:fold_start] + token_type_id_list[fold_end:]
                cur_type_index_in_token_ids_list_dev = type_index_in_token_ids_list[fold_start:fold_end]
                cur_type_index_in_token_ids_list_train = type_index_in_token_ids_list[
                                                         0:fold_start] + type_index_in_token_ids_list[fold_end:]

            train_data_index = [i for i in range(len(cur_data_list_train))]
            random.shuffle(train_data_index) # 打乱了顺序
            cur_data_list_train = [cur_data_list_train[index] for index in train_data_index] # 因为之前shuffle过，所以要这样
            cur_label_list_train = [cur_label_list_train[index] for index in train_data_index]
            cur_token_type_id_list_train = [cur_token_type_id_list_train[index] for index in train_data_index]
            cur_type_index_in_token_ids_list_train = [cur_type_index_in_token_ids_list_train[index] for index in
                                                      train_data_index]
            np.save("data/index_type_fold_data_{}/token_ids_train.npy".format(fold_index), cur_data_list_train)
            np.save("data/index_type_fold_data_{}/token_ids_dev.npy".format(fold_index), cur_data_list_dev)
            np.save("data/index_type_fold_data_{}/labels_train.npy".format(fold_index), cur_label_list_train)
            np.save("data/index_type_fold_data_{}/labels_dev.npy".format(fold_index), cur_label_list_dev)
            np.save("data/index_type_fold_data_{}/token_type_ids_train.npy".format(fold_index),
                    cur_token_type_id_list_train)
            np.save("data/index_type_fold_data_{}/token_type_ids_dev.npy".format(fold_index),
                    cur_token_type_id_list_dev)
            np.save("data/index_type_fold_data_{}/type_index_in_token_ids_train.npy".format(fold_index),
                    cur_type_index_in_token_ids_list_train)
            np.save("data/index_type_fold_data_{}/type_index_in_token_ids_dev.npy".format(fold_index),
                    cur_type_index_in_token_ids_list_dev)

    def parse_data_from_json(self, input_data):
        data_list = []
        label_list = []
        token_type_id_list = []
        type_index_in_token_ids_list = []
        all_event_type_split = []
        for i in range(65):
            event_type_str = self.id2labels_map.get(i)
            all_event_type_split.append(event_type_str)
        all_event_type_split = [ele.split("-")[-1] for ele in all_event_type_split]
        # all_event_type_split = [降价 结婚 晋级 ... ... ]

        # index_list = [i for i in range(len(input_data))]
        # random.shuffle(index_list)
        # for index in index_list:
        #     data = input_data[index]
        for data in input_data:
            sentence = data["text"]
            event_list = data["event_list"]
            event_type_label_list = [0] * self.labels_map_len
            type_index_in_token_ids = []
            text_len_for_event_raw_str = 0
            for event in event_list:
                event_type = event["event_type"]
                index_of_event_type = self.labels_map[event_type]
                event_type_label_list[index_of_event_type] = 1
            token_ids_org, token_type_ids = self.tokenizer.encode(sentence)
            token_ids = token_ids_org
            type_token_len = 0
            suffix_token_ids = []

            for index, event_type_raw in enumerate(all_event_type_split): # 包含56個
                event_type_token_ids = self.tokenizer.encode(event_type_raw)[0]
                text_len_for_event_raw_str += len(event_type_token_ids)

            text_allow_len = 510 - text_len_for_event_raw_str # 除去開始和結尾，510是最大長度，
            if len(token_ids) > text_allow_len:
                header_len = int(text_allow_len / 4)
                tail_len = text_allow_len - header_len
                dealt_token_ids = token_ids[1:-1]
                prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
                token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[-1]]
                token_type_ids = [0] * len(token_ids)

            for index, event_type_raw in enumerate(all_event_type_split): # 這裡把 event_type 的 token 放到text的token當中
                type_index_in_token_ids.append(len(token_ids))  # 這個句子text 的每個 event type token在 token化text 中的開始位置
                if index == 0:
                    event_type_token_ids = self.tokenizer.encode(event_type_raw)[0]
                    # [unused]
                    event_type_token_ids[0] = 5
                    type_token_len += len(event_type_token_ids)

                else:
                    event_type_token_ids = self.tokenizer.encode(event_type_raw)[0][1:]
                    type_token_len += len(event_type_token_ids)
                suffix_token_ids.extend(event_type_token_ids) # 後綴
                token_ids = token_ids + event_type_token_ids
                token_type_ids.extend([1] * len(event_type_token_ids))
            # if len(token_ids) > 512:
            #     text_allow_len = 510-type_token_len
            #     header_len = int(text_allow_len/4)
            #     tail_len = text_allow_len-header_len
            #     dealt_token_ids = token_ids_org[1:-1]
            #     prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids,header_len,tail_len,text_allow_len)
            #     prefix_token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[1]]
            #     token_ids = prefix_token_ids + suffix_token_ids
            #     token_type_ids = [0]*len(prefix_token_ids)+[1]*len(suffix_token_ids)
            #     print(len(token_ids))
            data_list.append(token_ids)
            token_type_id_list.append(token_type_ids)
            label_list.append(event_type_label_list)
            type_index_in_token_ids_list.append(type_index_in_token_ids)


        return data_list, token_type_id_list, label_list, type_index_in_token_ids_list

    def _read_json_file_kfold(self, train_input_file, eval_input_file, is_train):
        train_input_data = []
        with codecs.open(train_input_file, "r", encoding='utf-8') as f:
            for line in f:
                d_json = json.loads(line.strip())
                train_input_data.append(d_json)
        if is_train:
            with codecs.open(eval_input_file, "r", encoding='utf-8') as f:
                for line in f:
                    d_json = json.loads(line.strip())
                    train_input_data.append(d_json)
        data_list, token_type_ids_list, label_list, type_index_in_token_ids_list = self.parse_data_from_json(
            train_input_data)
        return data_list, token_type_ids_list, label_list, type_index_in_token_ids_list

    # def _read_json_file(self, train_input_file, eval_input_file, is_train):
    #     """_read_json_file"""
    #     train_input_data = []
    #     with codecs.open(train_input_file, "r", encoding='utf-8') as f:
    #         for line in f:
    #             d_json = json.loads(line.strip())
    #             train_input_data.append(d_json)
    #     if is_train:
    #         with codecs.open(eval_input_file, "r", encoding='utf-8') as f:
    #             for line in f:
    #                 d_json = json.loads(line.strip())
    #                 train_input_data.append(d_json)
    #     train_data_list, train_label_list, train_token_type_id_list, dev_data_list, dev_label_list, dev_token_type_id_list = self.parse_data_from_json(
    #         train_input_data)
    #     return train_data_list, train_label_list, train_token_type_id_list, dev_data_list, dev_label_list, dev_token_type_id_list

    def trans_single_data_for_test(self, text):
        token_ids_org, token_type_ids = self.tokenizer.encode(text)
        all_event_type_split = []
        for i in range(65):
            event_type_str = self.id2labels_map.get(i)
            all_event_type_split.append(event_type_str)
        all_event_type_split = [ele.split("-")[-1] for ele in all_event_type_split]
        token_ids = token_ids_org
        type_token_len = 0
        suffix_token_ids = []
        text_len_for_event_raw_str = 0
        type_index_in_token_ids = []
        for index, event_type_raw in enumerate(all_event_type_split):
            event_type_token_ids = self.tokenizer.encode(event_type_raw)[0]
            text_len_for_event_raw_str += len(event_type_token_ids)
        text_allow_len = 510 - text_len_for_event_raw_str
        if len(token_ids) > text_allow_len:
            header_len = int(text_allow_len / 4)
            tail_len = text_allow_len - header_len
            dealt_token_ids = token_ids[1:-1]
            prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
            token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[1]]
            token_type_ids = [0] * len(token_ids)

        for index, event_type_raw in enumerate(all_event_type_split):
            type_index_in_token_ids.append(len(token_ids))
            if index == 0:
                event_type_token_ids = self.tokenizer.encode(event_type_raw)[0]
                event_type_token_ids[0] = 5
                type_token_len += len(event_type_token_ids)
            else:
                event_type_token_ids = self.tokenizer.encode(event_type_raw)[0][1:]
                type_token_len += len(event_type_token_ids)
            suffix_token_ids.extend(event_type_token_ids)
            token_ids = token_ids + event_type_token_ids
            token_type_ids.extend([1] * len(event_type_token_ids))
        if len(token_ids) > 512:
            text_allow_len = 510 - type_token_len
            header_len = int(text_allow_len / 4)
            tail_len = text_allow_len - header_len
            dealt_token_ids = token_ids_org[1:-1]
            prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
            prefix_token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[1]]
            token_ids = prefix_token_ids + suffix_token_ids
            token_type_ids = [0] * len(prefix_token_ids) + [1] * len(suffix_token_ids)
        return token_ids, token_type_ids, type_index_in_token_ids


class EventRoleClassificationPrepare:
    def __init__(self, vocab_file, max_seq_length, all_role_type_event_file):
        self.max_seq_length = max_seq_length
        self.labels_map, self.id2labels_map = self.read_slot(all_role_type_event_file)
        self.labels_map_len = len(self.labels_map)
        self.tokenizer = Tokenizer(vocab_file, do_lower_case=False, )

    def read_slot(self, slot_file):
        labels_map = {}
        id2labels_map = {}
        with codecs.open(slot_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                arr = line.split("\t")
                labels_map[arr[0]] = int(arr[1])
                id2labels_map[int(arr[1])] = arr[0]
        return labels_map, id2labels_map

    def parse_data_from_json(self, input_data):
        data_list = []
        label_list = []
        token_type_id_list = []
        index_list = [i for i in range(len(input_data))]
        random.shuffle(index_list)
        for index in index_list:
            data = input_data[index]
            # for data in input_data:
            sentence = data["text"]
            sentence_token_ids, sentence_token_type_ids = self.tokenizer.encode(sentence)
            # sentence_token_type_ids = [ids+1 for ids in sentence_token_type_ids]
            event_list = data["event_list"]
            event_role_label_list = [0] * self.labels_map_len
            for event in event_list:
                event_type = event["event_type"]
                for index, role in enumerate(event["arguments"]):
                    role_type = role.get("role")
                    # event_role_type = event_type+"-"+role_type
                    index_of_event_role_type = self.labels_map[role_type]
                    event_role_label_list[index_of_event_role_type] = 1

            data_list.append(sentence_token_ids)
            token_type_id_list.append(sentence_token_type_ids)
            label_list.append(event_role_label_list)
        return data_list, token_type_id_list, label_list

    def _read_json_file(self, input_file):
        """_read_json_file"""
        input_data = []
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            for line in f:
                d_json = json.loads(line.strip())
                input_data.append(d_json)
        examples, examples_type_ids, examples_labels = self.parse_data_from_json(input_data)
        return examples, examples_type_ids, examples_labels

    def trans_single_data_for_test(self, text):
        sentence_token_ids, sentence_token_type_ids = self.tokenizer.encode(text)
        # sentence_chars = list(text)
        # sentence_chars = [w for w in sentence_chars if len(w.strip())!=0]
        # if len(sentence_chars) > self.max_seq_length - 2:
        #     sentence_chars = sentence_chars[0:(self.max_seq_length - 2)]
        # tokens = ["[CLS]"] + sentence_chars + ["[SEP]"]
        # token_type_ids = [0]*len(tokens)
        # token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return sentence_token_ids, sentence_token_type_ids


def event_data_generator_bert(input_Xs, labels):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        label = labels[index]
        yield (input_x, len(input_x)), label


def event_input_bert_fn(input_Xs, label_map_len, is_training, is_testing, args, input_Ys=None):
    _shapes = (([None], ()), [None])
    _types = ((tf.int32, tf.int32), tf.int32)
    _pads = ((0, 0), label_map_len - 1)
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_bert(input_Xs, input_Ys),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_data_generator_bert_mrc(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        start_y = start_Ys[index]
        end_y = end_Ys[index]
        token_type_id = token_type_ids[index]
        query_len = query_lens[index]
        yield (input_x, len(input_x), query_len, token_type_id), (start_y, end_y)


def event_input_bert_mrc_fn(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens, is_training, is_testing, args):
    _shapes = (([None], (), (), [None]), ([None], [None]))
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32), (tf.int32, tf.int32))
    _pads = ((0, 0, 0, 0), (0, 0))
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_bert_mrc(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
    if is_training:
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads, )
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_data_generator_bert_mrc_mul(input_Xs, Ys, token_type_ids, query_lens):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        y = Ys[index]
        # end_y = end_Ys[index]
        token_type_id = token_type_ids[index]
        query_len = query_lens[index]
        yield (input_x, len(input_x), query_len, token_type_id), y


def event_input_bert_mrc_mul_fn(input_Xs, Ys, token_type_ids, query_lens, is_training, is_testing, args):
    _shapes = (([None], (), (), [None]), [None])
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32)
    _pads = ((0, 0, 0, 0), 0)
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_bert_mrc_mul(input_Xs, Ys, token_type_ids, query_lens),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
    if is_training:
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads, )
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_data_generator_bert_class(input_Xs, token_type_ids, labels):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        label = labels[index]
        token_type_id = token_type_ids[index]
        yield (input_x, token_type_id, len(input_x)), label


def event_class_input_bert_fn(input_Xs, token_type_ids, label_map_len, is_training, is_testing, args, input_Ys=None):
    _shapes = (([None], [None], ()), [None])
    _types = ((tf.int32, tf.int32, tf.int32), tf.float32)
    _pads = ((0, 0, 0), 0.0)
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_bert_class(input_Xs, token_type_ids, input_Ys),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_data_generator_bert_binclass(input_Xs, token_type_ids, labels):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        label = labels[index]
        token_type_id = token_type_ids[index]
        yield (input_x, token_type_id, len(input_x)), label


def event_binclass_input_bert_fn(input_Xs, token_type_ids, label_map_len, is_training, is_testing, args, input_Ys=None):
    _shapes = (([None], [None], ()), [1])
    _types = ((tf.int32, tf.int32, tf.int32), tf.float32)
    _pads = ((0, 0, 0), 0.0)
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_bert_binclass(input_Xs, token_type_ids, input_Ys),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_index_data_generator_bert_class(input_Xs, token_type_ids, type_index_ids_list, labels):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        label = labels[index]
        token_type_id = token_type_ids[index]
        type_index_ids = type_index_ids_list[index]
        yield (input_x, token_type_id, len(input_x), type_index_ids), label


def event_index_class_input_bert_fn(input_Xs, token_type_ids, type_index_ids_list, label_map_len, is_training,
                                    is_testing, args, input_Ys=None):
    _shapes = (([None], [None], (), [label_map_len]), [None])
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32), tf.float32)
    _pads = ((0, 0, 0, 0), 0.0)
    ds = tf.data.Dataset.from_generator(
        lambda: event_index_data_generator_bert_class(input_Xs, token_type_ids, type_index_ids_list, input_Ys),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_data_generator_verify_mrc(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens, has_answer):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        start_y = start_Ys[index]
        end_y = end_Ys[index]
        token_type_id = token_type_ids[index]
        query_len = query_lens[index]
        has_answer_label = has_answer[index]
        yield (input_x, len(input_x), query_len, token_type_id), (start_y, end_y, has_answer_label)


def event_input_verfify_mrc_fn(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens, has_answer, is_training,
                               is_testing, args):
    _shapes = (([None], (), (), [None]), ([None], [None], [1]))
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32))
    _pads = ((0, 0, 0, 0), (0, 0, 0))
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_verify_mrc(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens, has_answer),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
    if is_training:
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads, )
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


if __name__ == "__main__":
    vocab_file_path = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("vocab_file"))
    # bert_config_file = os.path.join(event_config.get("bert_pretrained_model_path"), event_config.get("bert_config_path"))
    event_type_file = os.path.join(event_config.get("slot_list_root_path"), event_config.get("event_type_file"))
    data_loader = EventTypeClassificationPrepare(vocab_file_path, 512, event_type_file)
    train_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_train"))
    eval_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_eval"))
    # train_data_list,train_label_list,train_token_type_id_list,dev_data_list,dev_label_list,dev_token_type_id_list = data_loader._read_json_file(train_file,eval_file,is_train=True)
    train_data_list, train_label_list, train_token_type_id_list, dev_data_list, dev_label_list, dev_token_type_id_list = data_loader._merge_ee_and_re_datas(
        train_file, eval_file, "relation_extraction/data/train_data.json", "relation_extraction/data/dev_data.json")
