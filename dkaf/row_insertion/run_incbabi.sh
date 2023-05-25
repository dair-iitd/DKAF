DATALOC=`readlink -f $1`

ROW_INSERTION=$DATALOC/row_insertion/
MODELLOC='./runs/incbabi/'
name=incbabi

cd data_gen
mkdir -p $ROW_INSERTION
python generate.py --src_file $DATALOC/train.txt --tar_loc $ROW_INSERTION --dataset babi
python generate.py --src_file $DATALOC/dev.txt --tar_loc $ROW_INSERTION --dataset babi

cd ..
mkdir -p $MODELLOC
python -u run.py -dataset babi --data_loc=$ROW_INSERTION --dest $MODELLOC --use_ent_tags \
    -esize 200 -ehd 100 -bsz 32 -epochs 30 -clip 10 -lr 1e-4 -device cuda

python -u test.py -dataset babi --data_loc=$ROW_INSERTION --dest $MODELLOC --use_ent_tags \
    -esize 200 -ehd 100

cd data_gen
python augment.py --src_file $DATALOC/train.txt --tar_file $ROW_INSERTION/train.txt \
    --aug_file $ROW_INSERTION/train_infer_pred.json --dataset babi
