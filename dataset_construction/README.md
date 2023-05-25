# Dataset Construction
This package creates dialog datasets using by simulating evolving knowledge bases used in the paper.

## Requirements
Numpy, Pandas

## Usage
### 1. bAbI
To create incbAbI dataset from the paper run `bash run.sh` from the <i>babi</i> folder.
The command will create <i>incbabi</i> folder containing inconsistent training dialogs.
You can tweak the parameters of the simulation through command line arguments to `simulator.py`.
```
usage: simulator.py [-h] --data_loc DATA_LOC --dest_loc DEST_LOC --seed SEED --use_latest_kb {True,False} --start_date START_DATE --end_date END_DATE
                    --clock_resolution CLOCK_RESOLUTION

bAbI Evolving KB Simulator

optional arguments:
  -h, --help            show this help message and exit
  --data_loc DATA_LOC   Original bAbI data location
  --dest_loc DEST_LOC   Destination folder where simulated dialogs will be created
  --seed SEED           Seed value for the simulation
  --use_latest_kb {True,False}
                        Whether to use recent KB while saving the simulated dialogs. Setting this parameter to True will lead to inconsistent dialogs.
                        Otherwise, dialogs will have their contemporary KB.
  --start_date START_DATE
                        Start Date (yyyy-mm-dd) to be used for simulating evolving KB
  --end_date END_DATE   End Date (yyyy-mm-dd) to be used for simulating evolving KB
  --clock_resolution CLOCK_RESOLUTION
                        Clock Resolution (min) to be used for simulating evolving KB
```

### 2. BiTOD
To create incBiTOD dataset from the paper run `bash run.sh` from the <i>bitod</i> folder.
The command will create <i>incbitod</i> folder containing inconsistent training dialogs.

In case of BiTOD, we simulate evolving KB independently for each domain (hotel, restaurant, attraction). You can tweak the parameters of the simulation through command line arguments to `simulator_<domain>.py` where <domain> is one of hotel, restaurant or attraction. For example, in case of hotel, you can check the command line argument with `-h` option.
```
$ python simulator_hotel.py -h
usage: simulator_hotel.py [-h] -src_file SRC_FILE -kb_loc KB_LOC -tar_loc TAR_LOC [-ksz KBSIZE] [-rating_drop RATING_DROP] [-avail_prob AVAIL_PROB]
                          --use_latest_kb {True,False}

Hotel perturbation

optional arguments:
  -h, --help            show this help message and exit
  -src_file SRC_FILE, --src_file SRC_FILE
                        Hotel target file
  -kb_loc KB_LOC, --kb_loc KB_LOC
                        KB location
  -tar_loc TAR_LOC, --tar_loc TAR_LOC
                        Hotel target location
  -ksz KBSIZE, --kbsize KBSIZE
                        KB Size
  -rating_drop RATING_DROP, --rating_drop RATING_DROP
                        Rating Drop
  -avail_prob AVAIL_PROB, --avail_prob AVAIL_PROB
                        Hotel availibility probability
  --use_latest_kb {True,False}
                        Whether to use recent KB while saving the simulated dialogs. Setting this parameter to True will lead to inconsistent dialogs.
                        Otherwise, dialogs will have their contemporary KB.
```


## References
We thank authors of bAbI and BiTOD datasets

[1]: Bordes, Antoine and Jason Weston. “Learning End-to-End Goal-Oriented Dialog.” ArXiv abs/1605.07683 (2016): n. pag.

[2]: Lin, Zhaojiang et al. “BiToD: A Bilingual Multi-Domain Dataset For Task-Oriented Dialogue Modeling.” ArXiv abs/2106.02787 (2021): n. pag.
