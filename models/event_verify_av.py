import tensorflow as tf
import common_utils
import optimization
from models.tf_metrics import precision, recall, f1
# from albert import modeling,modeling_google
from bert import modeling
# from bert import modeling_theseus
from models.utils import focal_loss
from tensorflow.python.ops import metrics as metrics_lib

logger = common_utils.set_logger('NER Training...')


class VerfiyMRC(object):
    def __init__(self, params, bert_config):
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = 2
        self.bert_config = bert_config

    def __call__(self, input_ids, start_labels, end_labels, token_type_ids_list, query_len_list, text_length_list,
                 has_answer_label, is_training, is_testing=False):
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            token_type_ids=token_type_ids_list,
            use_one_hot_embeddings=False
        )
        bert_seq_output = bert_model.get_sequence_output()
        first_seq_hidden = bert_model.get_pooled_output()
        # bert_project = tf.layers.dense(bert_seq_output, self.hidden_units, activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project, rate=self.dropout_rate, training=is_training)
        start_logits = tf.layers.dense(bert_seq_output, self.num_labels)
        end_logits = tf.layers.dense(bert_seq_output, self.num_labels)
        query_span_mask = tf.cast(tf.sequence_mask(query_len_list), tf.int32)
        total_seq_mask = tf.cast(tf.sequence_mask(text_length_list), tf.int32)
        query_span_mask = query_span_mask * -1
        query_len_max = tf.shape(query_span_mask)[1]
        left_query_len_max = tf.shape(total_seq_mask)[1] - query_len_max
        zero_mask_left_span = tf.zeros((tf.shape(query_span_mask)[0], left_query_len_max), dtype=tf.int32)
        final_mask = tf.concat((query_span_mask, zero_mask_left_span), axis=-1)
        final_mask = final_mask + total_seq_mask
        predict_start_ids = tf.argmax(start_logits, axis=-1, name="pred_start_ids")
        predict_start_prob = tf.nn.softmax(start_logits, axis=-1)
        predict_end_prob = tf.nn.softmax(end_logits, axis=-1)
        predict_end_ids = tf.argmax(end_logits, axis=-1, name="pred_end_ids")
        # has_answer_logits = tf.layers.dropout(first_seq_hidden,rate=self.dropout_rate,training=is_training)
        has_answer_logits = tf.layers.dense(first_seq_hidden, 1)
        predict_has_answer_probs = tf.nn.sigmoid(has_answer_logits)
        if not is_testing:
            # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            # start_loss = ce_loss(start_logits,start_labels,final_mask,self.num_labels,True)
            # end_loss = ce_loss(end_logits,end_labels,final_mask,self.num_labels,True)

            # focal loss
            start_loss = focal_loss(start_logits, start_labels, final_mask, self.num_labels, True, 1.8)
            end_loss = focal_loss(end_logits, end_labels, final_mask, self.num_labels, True, 1.8)
            has_answer_label = tf.cast(has_answer_label, tf.float32)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=has_answer_label,
                                                                       logits=has_answer_logits)
            has_answer_loss = tf.reduce_mean(per_example_loss)
            # has_answer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=has_answer_logits))
            final_loss = (1.5 * start_loss + end_loss + has_answer_loss) / 3.0
            return final_loss, predict_start_ids, predict_end_ids, final_mask, predict_start_prob, predict_end_prob, predict_has_answer_probs
        else:
            return predict_start_ids, predict_end_ids, final_mask, predict_start_prob, predict_end_prob, predict_has_answer_probs


def event_verify_mrc_model_fn_builder(bert_config_file, init_checkpoints, args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'], features['text_length'], features['query_length'], features['token_type_ids']
        print(features)
        input_ids, text_length_list, query_length_list, token_type_id_list = features
        if labels is not None:
            start_labels, end_labels, has_answer_label = labels
        else:
            start_labels, end_labels, has_answer_label = None, None, None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tag_model = VerfiyMRC(params, bert_config)

        # input_ids,labels,token_type_ids_list,query_len_list,text_length_list,is_training,is_testing=False
        if is_testing:
            pred_start_ids, pred_end_ids, weight, predict_start_prob, predict_end_prob, has_answer_prob = tag_model(
                input_ids, start_labels, end_labels, token_type_id_list, query_length_list, text_length_list,
                has_answer_label, is_training, is_testing)
            # predict_ids,weight,predict_prob = tag_model(input_ids,labels,token_type_id_list,query_length_list,text_length_list,is_training,is_testing)
        else:
            loss, pred_start_ids, pred_end_ids, weight, predict_start_prob, predict_end_prob, has_answer_prob = tag_model(
                input_ids, start_labels, end_labels, token_type_id_list, query_length_list, text_length_list,
                has_answer_label, is_training)
            # loss,predict_ids,weight,predict_prob = tag_model(input_ids,labels,token_type_id_list,query_length_list,text_length_list,is_training,is_testing)

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
        output_spec = None
        # f1_score_val, f1_update_op_val = f1(labels=labels, predictions=pred_ids, num_classes=params["num_labels"],
        #                                     weights=weight)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, args.lr, params["train_steps"], params["num_warmup_steps"],
                                                     args.clip_norm)
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
            has_answer_pred = tf.where(has_answer_prob > 0.5, tf.ones_like(has_answer_prob),
                                       tf.zeros_like(has_answer_prob))

            # pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
            # weight = tf.sequence_mask(text_length_list)
            # precision_score, precision_update_op = precision(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            #
            # recall_score, recall_update_op =recall(labels=labels,
            #                                              predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            # def metric_fn(per_example_loss, label_ids, probabilities):

            #     logits_split = tf.split(probabilities, params["num_labels"], axis=-1)
            #     label_ids_split = tf.split(label_ids, params["num_labels"], axis=-1)
            #     # metrics change to auc of every class
            #     eval_dict = {}
            #     for j, logits in enumerate(logits_split):
            #         label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
            #         current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
            #         eval_dict[str(j)] = (current_auc, update_op_auc)
            #     eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
            #     return eval_dict
            # eval_metrics = metric_fn(per_example_loss, labels, pred_ids)
            f1_start_val, f1_update_op_val = f1(labels=start_labels, predictions=pred_start_ids, num_classes=2,
                                                weights=weight, average="macro")
            f1_end_val, f1_end_update_op_val = f1(labels=end_labels, predictions=pred_end_ids, num_classes=2,
                                                  weights=weight, average="macro")
            # f1_val,f1_update_op_val = f1(labels=labels,predictions=predict_ids,num_classes=3,weights=weight,average="macro")
            has_answer_label = tf.cast(has_answer_label, tf.float32)
            f1_has_val, f1_has_update_op_val = f1(labels=has_answer_label, predictions=has_answer_pred, num_classes=2)

            # f1_score_val_micro,f1_update_op_val_micro = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight,average="micro")

            # acc_score_val,acc_score_op_val = tf.metrics.accuracy(labels=labels,predictions=pred_ids,weights=weight)
            # eval_loss = tf.metrics.mean_squared_error(labels=labels, predictions=pred_ids,weights=weight)

            eval_metric_ops = {
                "f1_start_macro": (f1_start_val, f1_update_op_val),
                "f1_end_macro": (f1_end_val, f1_end_update_op_val),
                "f1_has_answer_macro": (f1_has_val, f1_has_update_op_val),
                "eval_loss": tf.metrics.mean(values=loss)}

            # eval_metric_ops = {
            # "f1_macro":(f1_val,f1_update_op_val),
            # "eval_loss":tf.metrics.mean(values=loss)}

            # eval_hook_dict = {"f1":f1_score_val,"loss":loss}

            # eval_logging_hook = tf.train.LoggingTensorHook(
            #     at_end=True,every_n_iter=args.print_log_steps)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metric_ops,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"start_ids": pred_start_ids, "end_ids": pred_end_ids, "start_probs": predict_start_prob,
                             "end_probs": predict_end_prob, "has_answer_probs": has_answer_prob}
            )
            # output_spec = tf.estimator.EstimatorSpec(
            #     mode=mode,
            #     predictions={"pred_ids":predict_ids,"pred_probs":predict_prob}
            # )
        return output_spec

    return model_fn
