DATALOC=`readlink -f $1`

ROW_INSERTION=$DATALOC/row_insertion/
REWARD_FN=$DATALOC/reward_fn/
ROW_DELETION=$DATALOC/row_deletion/
MODELLOC='./runs/incbabi/'
name=incbabi

cd data_gen
mkdir -p $ROW_DELETION
python generate.py --src_file $ROW_INSERTION/train.txt --tar_file $ROW_DELETION/train_rd.json --dataset babi

cd ..
python -u prepare_reward_table.py -dataset babi -data $ROW_DELETION -rw_loc $REWARD_FN -dest $ROW_DELETION --device cuda
python -u run.py -dataset babi -data $ROW_DELETION -dest $MODELLOC -rw_loc $ROW_DELETION --mem_hops 8 \
      -device cuda --use_mapo --num_epoch 200 --batch_size 32 --use_ent_tags \
      --sample_size 2 -mapo_update_freq 4 -dr 0.0 --clip 10

python -u test.py -dataset babi -data $ROW_DELETION -dest $MODELLOC -rw_loc $ROW_DELETION --mem_hops 8 \
    -device cuda --use_mapo --num_epoch 32 --use_ent_tags \
    --batch_size 32 --sample_size 2 -dr 0.0

cd data_gen
python augment.py --src_file $ROW_INSERTION/train.txt \
    --tar_file $ROW_DELETION/train.txt \
    --aug_file $ROW_DELETION/train_rd_pred.json --dataset babi
