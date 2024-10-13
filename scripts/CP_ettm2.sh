if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ChannelPrediction" ]; then
    mkdir ./logs/ChannelPrediction
fi
seq_len=336
pred_len=$seq_len
model_name=TMCN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2024
for ch in {0..6}
do
    python -u CP_run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 128 --learning_rate 0.0001\
      --channel $ch\
      >logs/ChannelPrediction/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$ch.log 
done