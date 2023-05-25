DATALOC=`readlink -f $1`

ROW_INSERTION=$DATALOC/row_insertion/
REWARD_FN=$DATALOC/reward_fn/
ROW_DELETION=$DATALOC/row_deletion/
ROW_COMPLETION=$DATALOC/row_completion/
MODELLOC='./runs/incbabi/'
name=incbabi

cd data_gen
mkdir -p $ROW_COMPLETION
python generate.py --src_file $ROW_DELETION/train.txt --tar_loc $ROW_COMPLETION --dataset babi

cd ..
python -u prepare_reward_table.py -dataset babi -data $ROW_COMPLETION -rw_loc $REWARD_FN -dest $ROW_COMPLETION --device cuda
python -u run.py -dataset babi -data $ROW_COMPLETION -dest $MODELLOC -rw_loc $ROW_COMPLETION --mem_hops 8 \
    -device cuda --use_mapo --num_epoch 200 --batch_size 32 -dr 0.0 \
    --sample_size 4 -mapo_update_freq 4 --clip 10 -buff_clip 0.5 --use_ent_tags

python -u test.py -dataset babi -data $ROW_COMPLETION -dest $MODELLOC -rw_loc $ROW_COMPLETION --mem_hops 8 \
    -device cuda --use_mapo --num_epoch 32 --batch_size 32 --sample_size 8 -dr 0.0 --use_ent_tags

cd data_gen
python augment.py --src_file $ROW_DELETION/train.txt --tar_file $ROW_COMPLETION/train.txt \
    --aug_file $ROW_COMPLETION/train_infer_pred.json --dataset babi
