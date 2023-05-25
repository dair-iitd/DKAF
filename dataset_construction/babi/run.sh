DEST_LOC=./incbabi
python simulator.py --data_loc ./orig_data/ \
    --dest_loc $DEST_LOC \
    --seed 42 \
    --use_latest_kb True \
    --start_date 2021-10-01 \
    --end_date 2021-12-31 \
    --clock_resolution 30

cp ./orig_data/dialog-babi-task5-full-dialogs-dev.txt $DEST_LOC/dev.txt
cp ./orig_data/dialog-babi-task5-full-dialogs-tst.txt $DEST_LOC/test.txt
cp ./orig_data/dialog-babi-kb-all.txt $DEST_LOC/kb.txt
