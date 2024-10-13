if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ChannelPrediction" ]; then
    mkdir ./logs/ChannelPrediction
fi
seq_len=48
pred_len=$seq_len
model_name=TMCN

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 200\
      --lradj 'constant'\
      --itr 1 --channel $ch\
      --learning_rate 0.0005\
      >logs/ChannelPrediction/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$ch.log 
done