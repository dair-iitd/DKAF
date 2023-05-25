DATALOC=`readlink -f $1`

ROW_INSERTION=$DATALOC/row_insertion/
MODELLOC='./runs/incbitod/'
name=incbitod

cd data_gen
mkdir -p $ROW_INSERTION
python generate.py --src_file $DATALOC/train.json --tar_loc $ROW_INSERTION --dataset bitod
python generate.py --src_file $DATALOC/val.json --tar_loc $ROW_INSERTION --dataset bitod
mv $ROW_INSERTION/val_gold.json $ROW_INSERTION/dev_gold.json

cd ..
mkdir -p $MODELLOC
python -u run.py -dataset bitod --data_loc=$ROW_INSERTION --dest $MODELLOC --use_ent_tags \
    -esize 200 -ehd 100 -bsz 32 -epochs 30 -clip 10 -lr 1e-4 -device cuda

python -u test.py -dataset bitod --data_loc=$ROW_INSERTION --dest $MODELLOC --use_ent_tags \
    -esize 200 -ehd 100

cd data_gen
python augment.py --src_file $DATALOC/train.json --tar_file $ROW_INSERTION/train.json \
    --aug_file $ROW_INSERTION/train_infer_pred.json --dataset bitod
