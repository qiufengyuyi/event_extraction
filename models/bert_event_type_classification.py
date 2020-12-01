import tensorflow as tf
import numpy as np
import common_utils
# import optimization_hvd
import optimization
from models.tf_metrics import precision, recall, f1
from bert import modeling

logger = common_utils.set_logger('NER Training...')


class bertEventType(object):
    def __init__(self, params, bert_config):
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.rnn_size = params["rnn_size"]
        self.num_layers = params["num_layers"]
        self.hidden_units = params["hidden_units"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config

    def __call__(self, input_ids, labels, text_length_list, token_type_ids, is_training, is_testing=False):

        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        bert_output = bert_model.get_pooled_output()
        bert_project = tf.layers.dense(bert_output, self.num_labels)
        pred_prob = tf.nn.sigmoid(bert_project, name="pred_probs")
        # print(labels)
        # lstm_layer = BLSTM(None, self.rnn_size, self.num_layers, 1.-self.dropout_rate,
        #                    lengths=text_length_list, is_training=is_training)
        # lstm_output = lstm_layer.blstm_layer(bert_embedding)

        # lstm_crf_model = BLSTM_CRF(bert_embedding,self.hidden_units,self.rnn_size,self.num_layers,self.dropout_rate,self.num_labels,labels,text_length_list,is_training)
        # loss, logits, trans, pred_ids = lstm_crf_model.add_blstm_crf_layer()

        # bert_project = tf.layers.dense(bert_embedding, self.hidden_units,activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project,rate=self.dropout_rate,training=is_training)
        # bert_project = tf.layers.dense(bert_project, self.num_labels)
        # pred_ids = tf.argmax(bert_project, axis=-1, name="pred_ids")
        # pred_prob = tf.nn.softmax(bert_project, axis=-1, name="pred_probs")
        if not is_testing:
            # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            # log_probs = tf.nn.log_softmax(bert_project, axis=-1)
            # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # per_example_loss = per_example_loss * weight
            # loss = tf.reduce_sum(per_example_loss,axis=-1)
            # loss = tf.reduce_mean(loss)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=bert_project)
            per_example_loss *= self.class_weight
            loss = tf.reduce_mean(per_example_loss)
            # loss = tf.reduce_mean(per_example_loss)
            # loss = dice_dsc_loss(bert_project,labels,text_length_list,weight,self.num_labels)
            # loss = focal_dsc_loss(bert_project,labels,text_length_list,weight,self.num_labels)
            return per_example_loss, loss, pred_prob
        else:
            return pred_prob


class bertEventTypeModified(object):
    def __init__(self, params, bert_config):
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.rnn_size = params["rnn_size"]
        self.num_layers = params["num_layers"]
        self.hidden_units = params["hidden_units"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config

    def __call__(self, input_ids, labels, text_length_list, token_type_ids, type_index_in_token_ids, is_training,
                 is_testing=False):

        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        bert_embedding = bert_model.get_sequence_output()
        # bert_embedding = tf.layers.dropout(bert_embedding,rate=0.2,training=is_training)
        batch_ids = tf.range(0, tf.shape(bert_embedding)[0])
        batch_ids = tf.expand_dims(batch_ids, 1)
        batch_ids = tf.expand_dims(batch_ids, -1)
        batch_ids = tf.tile(batch_ids, [1, self.num_labels, 1])
        type_index_in_token_ids = tf.expand_dims(type_index_in_token_ids, axis=-1) # 变成 (batch_size, 65, 1)
        type_index = tf.concat([batch_ids, type_index_in_token_ids], axis=-1)
        type_head_tensor = tf.gather_nd(bert_embedding, type_index)
        print(type_head_tensor)
        # bert_output = bert_model.get_pooled_output()
        # type_head_tensor = tf.layers.dense(type_head_tensor,768,activation=tf.nn.relu)
        # type_head_tensor = tf.layers.dropout(type_head_tensor,rate=0.2,training=is_training)
        bert_project = tf.layers.dense(type_head_tensor, 1)

        pred_prob = tf.nn.sigmoid(bert_project, name="pred_probs")
        pred_prob = tf.squeeze(pred_prob, axis=-1)
        # print(pred_prob)
        # print(labels)
        # lstm_layer = BLSTM(None, self.rnn_size, self.num_layers, 1.-self.dropout_rate,
        #                    lengths=text_length_list, is_training=is_training)
        # lstm_output = lstm_layer.blstm_layer(bert_embedding)

        # lstm_crf_model = BLSTM_CRF(bert_embedding,self.hidden_units,self.rnn_size,self.num_layers,self.dropout_rate,self.num_labels,labels,text_length_list,is_training)
        # loss, logits, trans, pred_ids = lstm_crf_model.add_blstm_crf_layer()

        # bert_project = tf.layers.dense(bert_embedding, self.hidden_units,activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project,rate=self.dropout_rate,training=is_training)
        # bert_project = tf.layers.dense(bert_project, self.num_labels)
        # pred_ids = tf.argmax(bert_project, axis=-1, name="pred_ids")

        # pred_prob = tf.nn.softmax(bert_project, axis=-1, name="pred_probs")

        if not is_testing:
            labels = tf.expand_dims(labels, axis=-1)
            # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            # log_probs = tf.nn.log_softmax(bert_project, axis=-1)
            # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # per_example_loss = per_example_loss * weight
            # loss = tf.reduce_sum(per_example_loss,axis=-1)
            # loss = tf.reduce_mean(loss)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=bert_project)
            per_example_loss *= self.class_weight
            loss = tf.reduce_mean(per_example_loss)

            # loss = tf.reduce_mean(per_example_loss)
            # loss = dice_dsc_loss(bert_project,labels,text_length_list,weight,self.num_labels)
            # loss = focal_dsc_loss(bert_project,labels,text_length_list,weight,self.num_labels)

            return per_example_loss, loss, pred_prob
        else:
            return pred_prob


def bert_classification_model_fn_builder(bert_config_file, init_checkpoints, args): # bert_config.json 文件路径， checkpoints 文件路径
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'], features['token_type_ids'], features['text_length'], features[
                'type_index_in_ids_list']
        #         print(features)
        # input_ids,token_type_ids,text_length_list = features
        input_ids, token_type_ids, text_length_list, type_index_in_ids_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        # tag_model = bertEventType(params,bert_config)
        tag_model = bertEventTypeModified(params, bert_config)
        if is_testing:
            # pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids,is_training,is_testing)
            pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids, type_index_in_ids_list,
                                 is_training, is_testing)
        else:
            per_example_loss, loss, pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids,
                                                         type_index_in_ids_list, is_training)

        # def metric_fn(label_ids, pred_ids):
        #     return {
        #         'precision': precision(label_ids, pred_ids, params["num_labels"]),
        #         'recall': recall(label_ids, pred_ids, params["num_labels"]),
        #         'f1': f1(label_ids, pred_ids, params["num_labels"])
        #     }
        #
        # eval_metrics = metric_fn(labels, pred_ids)
        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None # 占位 output spec
        # f1_score_val, f1_update_op_val = f1(labels=labels, predictions=pred_ids, num_classes=params["num_labels"],
        #                                     weights=weight)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, args.lr, params["decay_steps"], None, False)
            hook_dict = {}
            # precision_score, precision_update_op = precision(labels=labels, predictions=pred_ids,
            #                                                  num_classes=params["num_labels"], weights=weight)
            #
            # recall_score, recall_update_op = recall(labels=labels,
            #                                         predictions=pred_ids, num_classes=params["num_labels"],
            #                                         weights=weight)
            hook_dict['loss'] = loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.print_log_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            
            # 全部转笔乘0 1
            pred_ids = tf.where(pred_ids > 0.5, tf.ones_like(pred_ids), tf.zeros_like(pred_ids)) # Return the elements, either from x or y, depending on the condition

            def metric_fn(per_example_loss, label_ids, probabilities):

                logits_split = tf.split(probabilities, params["num_labels"], axis=-1)
                label_ids_split = tf.split(label_ids, params["num_labels"], axis=-1)
                # split 为 （batch_size, 1）
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split): # 表示每个事件类别的概率输出
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict

            eval_metrics = metric_fn(per_example_loss, labels, pred_ids)
            # pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
            # weight = tf.sequence_mask(text_length_list)
            # precision_score, precision_update_op = precision(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            #
            # recall_score, recall_update_op =recall(labels=labels,
            #                                              predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            # f1_score_val,f1_update_op_val = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight,average="macro")
            # f1_score_val_micro,f1_update_op_val_micro = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],average="macro")

            # acc_score_val,acc_score_op_val = tf.metrics.accuracy(labels=labels,predictions=pred_ids,weights=weight)
            # eval_loss = tf.metrics.mean_squared_error(labels=labels, predictions=pred_ids,weights=weight)
            # eval_metric_ops = {
            # "f1_score_micro":(f1_score_val_micro,f1_update_op_val_micro)}
            # eval_hook_dict = {"f1":f1_score_val,"loss":loss}

            # eval_logging_hook = tf.train.LoggingTensorHook(
            #     at_end=True,every_n_iter=args.print_log_steps)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metrics,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


def bert_binaryclassification_model_fn_builder(bert_config_file, init_checkpoints, args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'], features['token_type_ids'], features['text_length']
        #         print(features)
        # input_ids,token_type_ids,text_length_list = features
        input_ids, token_type_ids, text_length_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        # tag_model = bertEventType(params,bert_config)
        tag_model = bertEventType(params, bert_config)
        if is_testing:
            # pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids,is_training,is_testing)
            pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids, is_training, is_testing)
        else:
            per_example_loss, loss, pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids,
                                                         is_training)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, args.lr, params["decay_steps"], None, False)
            hook_dict = {}

            hook_dict['loss'] = loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.print_log_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            pred_ids = tf.where(pred_ids > 0.5, tf.ones_like(pred_ids), tf.zeros_like(pred_ids))
            print(pred_ids)
            print(labels)
            f1_score_val_micro, f1_update_op_val_micro = f1(labels=labels, predictions=pred_ids, num_classes=2)
            eval_metrics = {"f1_score_micro": (f1_score_val_micro, f1_update_op_val_micro)}
            eval_metrics['eval_loss'] = tf.metrics.mean(values=per_example_loss)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metrics,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn
