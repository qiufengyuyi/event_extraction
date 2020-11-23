for((index=2; index < 6; index++))
do
    echo ">> traning fold ${index}"
    python run_event.py --model_type classification --fold_index ${index} --dropout_prob 0.2 --epochs 5 --lr 2e-6 --clip_norm 5.0 --train_batch_size 2 --valid_batch_size 4 --shuffle_buffer 128 --do_train --do_test --tolerant_steps 500 --run_hook_steps 50 --print_log_steps 50 --decay_epoch 10 --pre_buffer_size 16 --bert_used --gpu_nums 2 --model_checkpoint_dir type_class_bert_model_dir --model_pb_dir type_class_bert_model_pb
done