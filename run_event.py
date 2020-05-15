import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from argparse import ArgumentParser
from train_helper import run_event_role_mrc, run_event_classification
from train_helper import run_event_binclassification, run_event_verify_role_mrc
import numpy as np

np.set_printoptions(threshold=np.inf)
tf.logging.set_verbosity(tf.logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="role", type=str)
    parser.add_argument("--dropout_prob", default=0.2, type=float)
    parser.add_argument("--rnn_units", default=256, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    # bert lr
    parser.add_argument("--lr", default=1e-5, type=float)
    # parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--clip_norm", default=5.0, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--shuffle_buffer", default=128, type=int)
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_test", action='store_true', default=True)
    parser.add_argument("--gen_new_data", action='store_true', default=False)
    parser.add_argument("--tolerant_steps", default=200, type=int)
    parser.add_argument("--run_hook_steps", default=100, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--hidden_units", default=128, type=int)
    parser.add_argument("--print_log_steps", default=50, type=int)
    parser.add_argument("--decay_epoch", default=12, type=int)
    parser.add_argument("--pre_buffer_size", default=1, type=int)
    parser.add_argument("--bert_used", default=False, action='store_true')
    parser.add_argument("--gpu_nums", default=1, type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="role_bert_model_dir")
    parser.add_argument("--model_pb_dir", type=str, default="role_bert_model_pb")
    parser.add_argument("--fold_index", type=int)

    args = parser.parse_args()
    if args.model_type == "role":
        run_event_role_mrc(args)
    elif args.model_type == "classification":
        run_event_classification(args)
    elif args.model_type == "binary":
        run_event_binclassification(args)
    elif args.model_type == "avmrc":
        run_event_verify_role_mrc(args)
    # if args.bert_used:
    #     if args.model_type == "bert_mrc":
    #         if args.theseus_compressed:
    #             print(args.model_type)
    #             run_bert_mrc_theseus(args)
    #         else:
    #             run_bert_mrc(args)
    #     else:
    #         run_bert(args)
    # else:
    #     if args.model_type == "lstm_crf" or args.model_type == "lstm_only":
    #         run_train(args)
    #     elif args.model_type=="lstm_cnn_crf":
    #         run_train_cnn(args)
    #     else:
    #         run_lan(args)
    # run_event_trigger_bert(args)


if __name__ == '__main__':
    main()
