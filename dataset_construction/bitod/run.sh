# Simulate KB for each domain in BiTOD independently.
DEST_LOC=./incbitod/
mkdir $DEST_LOC
python simulator_hotel.py -src_file orig_data/train.json --kb_loc orig_data/ --tar_loc $DEST_LOC --avail_prob 0.5 --use_latest_kb True
python simulator_hotel.py -src_file orig_data/val.json --kb_loc orig_data/ --tar_loc $DEST_LOC --avail_prob 0.5 --use_latest_kb False
python simulator_hotel.py -src_file orig_data/test.json --kb_loc orig_data/ --tar_loc $DEST_LOC --avail_prob 0.5 --use_latest_kb False

# python simulator_attraction.py -src_file orig_data/train.json --kb_loc orig_data/ --tar_loc $DEST_LOC --avail_prob 1.0 --use_latest_kb True
# python simulator_attraction.py -src_file orig_data/val.json --kb_loc orig_data/ --tar_loc $DEST_LOC --avail_prob 1.0 --use_latest_kb False
# python simulator_attraction.py -src_file orig_data/test.json --kb_loc orig_data/ --tar_loc $DEST_LOC --avail_prob 1.0 --use_latest_kb False

python simulator_restaurant.py -src_file orig_data/train.json --kb_loc orig_data/ --tar_loc $DEST_LOC --start_date 2021-10-01 --end_date 2022-06-30 -clock_resolution 30 --use_latest_kb True
python simulator_restaurant.py -src_file orig_data/val.json --kb_loc orig_data/ --tar_loc $DEST_LOC --start_date 2021-10-01 --end_date 2022-06-30 -clock_resolution 30 --use_latest_kb False
python simulator_restaurant.py -src_file orig_data/test.json --kb_loc orig_data/ --tar_loc $DEST_LOC --start_date 2021-10-01 --end_date 2022-06-30 -clock_resolution 30 --use_latest_kb False

# Combine all the datasets.
python combine.py $DEST_LOC/train_hotels.json \
    $DEST_LOC/train_attraction.json \
    $DEST_LOC/train_restaurant.json \
    $DEST_LOC/train.json

python combine.py $DEST_LOC/val_hotels.json \
    $DEST_LOC/val_attraction.json \
    $DEST_LOC/val_restaurant.json \
    $DEST_LOC/val.json

python combine.py $DEST_LOC/test_hotels.json \
    $DEST_LOC/test_attraction.json \
    $DEST_LOC/test_restaurant.json \
    $DEST_LOC/test.json

# Clean up
rm $DEST_LOC/train_hotels.json $DEST_LOC/train_attraction.json $DEST_LOC/train_restaurant.json
rm $DEST_LOC/val_hotels.json $DEST_LOC/val_attraction.json $DEST_LOC/val_restaurant.json
rm $DEST_LOC/test_hotels.json $DEST_LOC/test_attraction.json $DEST_LOC/test_restaurant.json

python gen_entities.py --src_loc $DEST_LOC/
