DATALOC=`readlink -f $1`

REWARD_FN=$DATALOC/reward_fn/
MODELLOC='./runs/incbabi/'
name=incbabi

cd data_gen

mkdir -p $REWARD_FN
python generate.py --src_file $DATALOC/dev.txt --tar_file $REWARD_FN/dev.json --dataset babi
python generate.py --src_file $DATALOC/row_insertion/train.txt --tar_file $REWARD_FN/train.json --dataset babi

cd ..
mkdir -p $MODELLOC
python -u run.py -dataset babi --data_loc=$REWARD_FN --dest $MODELLOC --use_ent_tags \
    -esize 200 -ehd 100 -bsz 32 -epochs 100 -clip 10 -lr 1e-4 \
    --mem_hops 8 -dr 0.0 --use_past_only --seed 42

python -u test.py -dataset babi --data_loc=$REWARD_FN --dest $MODELLOC --use_ent_tags \
    -esize 200 -ehd 100 -bsz 32 -epochs 80 -clip 10 -lr 1e-4 --mem_hops 8 --use_past_only -dr 0.0
