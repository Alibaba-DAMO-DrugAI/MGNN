#! /bin/bash
ROOT=$(dirname dirname "$0")
cd "$ROOT"
cd "mgnn"

python kfold_2016.py --dataset 2016_exp_eta2_n56 --fold 0
python kfold_2016.py --dataset 2016_exp_eta2_n56 --fold 1
python kfold_2016.py --dataset 2016_exp_eta2_n56 --fold 2
python kfold_2016.py --dataset 2016_exp_eta2_n56 --fold 3
python kfold_2016.py --dataset 2016_exp_eta2_n56 --fold 4
python kfold_2016.py --dataset 2016_exp_eta5_n56 --fold 0
python kfold_2016.py --dataset 2016_exp_eta5_n56 --fold 1
python kfold_2016.py --dataset 2016_exp_eta5_n56 --fold 2
python kfold_2016.py --dataset 2016_exp_eta5_n56 --fold 3
python kfold_2016.py --dataset 2016_exp_eta5_n56 --fold 4
python kfold_2016.py --dataset 2016_exp_eta10_n56 --fold 0
python kfold_2016.py --dataset 2016_exp_eta10_n56 --fold 1
python kfold_2016.py --dataset 2016_exp_eta10_n56 --fold 2
python kfold_2016.py --dataset 2016_exp_eta10_n56 --fold 3
python kfold_2016.py --dataset 2016_exp_eta10_n56 --fold 4
python kfold_2016.py --dataset 2016_exp_eta20_n56 --fold 0
python kfold_2016.py --dataset 2016_exp_eta20_n56 --fold 1
python kfold_2016.py --dataset 2016_exp_eta20_n56 --fold 2
python kfold_2016.py --dataset 2016_exp_eta20_n56 --fold 3
python kfold_2016.py --dataset 2016_exp_eta20_n56 --fold 4
python kfold_2016.py --dataset 2016_lor_eta2_n56 --fold 0
python kfold_2016.py --dataset 2016_lor_eta2_n56 --fold 1
python kfold_2016.py --dataset 2016_lor_eta2_n56 --fold 2
python kfold_2016.py --dataset 2016_lor_eta2_n56 --fold 3
python kfold_2016.py --dataset 2016_lor_eta2_n56 --fold 4
python kfold_2016.py --dataset 2016_lor_eta5_n56 --fold 0
python kfold_2016.py --dataset 2016_lor_eta5_n56 --fold 1
python kfold_2016.py --dataset 2016_lor_eta5_n56 --fold 2
python kfold_2016.py --dataset 2016_lor_eta5_n56 --fold 3
python kfold_2016.py --dataset 2016_lor_eta5_n56 --fold 4
python kfold_2016.py --dataset 2016_lor_eta10_n56 --fold 0
python kfold_2016.py --dataset 2016_lor_eta10_n56 --fold 1
python kfold_2016.py --dataset 2016_lor_eta10_n56 --fold 2
python kfold_2016.py --dataset 2016_lor_eta10_n56 --fold 3
python kfold_2016.py --dataset 2016_lor_eta10_n56 --fold 4
python kfold_2016.py --dataset 2016_lor_eta20_n56 --fold 0
python kfold_2016.py --dataset 2016_lor_eta20_n56 --fold 1
python kfold_2016.py --dataset 2016_lor_eta20_n56 --fold 2
python kfold_2016.py --dataset 2016_lor_eta20_n56 --fold 3
python kfold_2016.py --dataset 2016_lor_eta20_n56 --fold 4
python kfold_2016.py --dataset 2016_56_dist --fold 0
python kfold_2016.py --dataset 2016_56_dist --fold 1
python kfold_2016.py --dataset 2016_56_dist --fold 2
python kfold_2016.py --dataset 2016_56_dist --fold 3
python kfold_2016.py --dataset 2016_56_dist --fold 4
python kfold_2016.py --dataset 2016_56_elec --fold 0
python kfold_2016.py --dataset 2016_56_elec --fold 1
python kfold_2016.py --dataset 2016_56_elec --fold 2
python kfold_2016.py --dataset 2016_56_elec --fold 3
python kfold_2016.py --dataset 2016_56_elec --fold 4