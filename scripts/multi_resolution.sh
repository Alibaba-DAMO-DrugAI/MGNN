#! /bin/bash
ROOT=$(dirname dirname "$0")
cd "$ROOT"
cd "data"
pwd

# PDBbind2007
python dataset.py --name='PDBbind2007' --exponential --eta=2
python dataset.py --name='PDBbind2007' --exponential --eta=5
python dataset.py --name='PDBbind2007' --exponential --eta=10
python dataset.py --name='PDBbind2007' --exponential --eta=20
python dataset.py --name='PDBbind2007' --lorentz --eta=2
python dataset.py --name='PDBbind2007' --lorentz --eta=5
python dataset.py --name='PDBbind2007' --lorentz --eta=10
python dataset.py --name='PDBbind2007' --lorentz --eta=20
python dataset.py --name='PDBbind2007'

# PDBbind2013
python dataset.py --name='PDBbind2013' --exponential --eta=2
python dataset.py --name='PDBbind2013' --exponential --eta=5
python dataset.py --name='PDBbind2013' --exponential --eta=10
python dataset.py --name='PDBbind2013' --exponential --eta=20
python dataset.py --name='PDBbind2013' --lorentz --eta=2
python dataset.py --name='PDBbind2013' --lorentz --eta=5
python dataset.py --name='PDBbind2013' --lorentz --eta=10
python dataset.py --name='PDBbind2013' --lorentz --eta=20
python dataset.py --name='PDBbind2013'

# PDBbind2016
python dataset.py --name='PDBbind2016' --exponential --eta=2
python dataset.py --name='PDBbind2016' --exponential --eta=5
python dataset.py --name='PDBbind2016' --exponential --eta=10
python dataset.py --name='PDBbind2016' --exponential --eta=20
python dataset.py --name='PDBbind2016' --lorentz --eta=2
python dataset.py --name='PDBbind2016' --lorentz --eta=5
python dataset.py --name='PDBbind2016' --lorentz --eta=10
python dataset.py --name='PDBbind2016' --lorentz --eta=20
python dataset.py --name='PDBbind2016'

# PDBbind2019
python dataset.py --name='PDBbind2019' --exponential --eta=2
python dataset.py --name='PDBbind2019' --exponential --eta=5
python dataset.py --name='PDBbind2019' --exponential --eta=10
python dataset.py --name='PDBbind2019' --exponential --eta=20
python dataset.py --name='PDBbind2019' --lorentz --eta=2
python dataset.py --name='PDBbind2019' --lorentz --eta=5
python dataset.py --name='PDBbind2019' --lorentz --eta=10
python dataset.py --name='PDBbind2019' --lorentz --eta=20
python dataset.py --name='PDBbind2019'

# SARS-CoV-BA
python dataset.py --name='SARS-CoV-BA' --exponential --eta=2
python dataset.py --name='SARS-CoV-BA' --exponential --eta=5
python dataset.py --name='SARS-CoV-BA' --exponential --eta=10
python dataset.py --name='SARS-CoV-BA' --exponential --eta=20
python dataset.py --name='SARS-CoV-BA' --lorentz --eta=2
python dataset.py --name='SARS-CoV-BA' --lorentz --eta=5
python dataset.py --name='SARS-CoV-BA' --lorentz --eta=10
python dataset.py --name='SARS-CoV-BA' --lorentz --eta=20
python dataset.py --name='SARS-CoV-BA'

# Pretrain
python pretrain_dataset.py --name='PDBbind2007'
python pretrain_dataset.py --name='PDBbind2013'
python pretrain_dataset.py --name='PDBbind2016'