#! /bin/bash
ROOT=$(dirname dirname "$0")
cd "$ROOT"
cd "mgnn"

# PDBbind2007
python pretrain.py --dataset 2007_exp_eta20_n56_pretrain --outfile 2007_exp_eta20_n56_pretrain
python pretrain.py --dataset 2007_56_elec_pretrain --outfile 2007_56_elec_pretrain

# PDBbind2013
python pretrain.py --dataset 2013_exp_eta20_n56_pretrain --outfile 2013_exp_eta20_n56_pretrain
python pretrain.py --dataset 2013_56_elec_pretrain --outfile 2013_56_elec_pretrain

# PDBbind2016
python pretrain.py --dataset 2016_exp_eta20_n56_pretrain --outfile 2016_exp_eta20_n56_pretrain
python pretrain.py --dataset 2016_56_elec_pretrain --outfile 2016_56_elec_pretrain