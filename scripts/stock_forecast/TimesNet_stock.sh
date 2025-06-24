if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/classification" ]; then
    mkdir ./logs/classification
fi

# You can change the model_name and other parameters as needed
model_name=TimesNet
seq_len=96

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /kaggle/input/binary/ \
  --model_id stock_$seq_len \
  --model $model_name \
  --data stock \
  --train_file train.parquet \
  --val_file val.parquet \
  --test_file test.parquet \
  --label_column label \
  --group_column stock_id \
  --datetime_column datetime \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --label_len 0 \
  --c_out 2 \
  --use_gpu True \
  --gpu 0 \
  --e_layers 3 \
  --d_model 32 \
  --d_ff 128 \
  --top_k 3 \
  --batch_size 512 \
  --learning_rate 0.001 \
  --train_epochs 500 \
  --patience 10 \
  --des 'Stock_Exp' \
  --itr 1
