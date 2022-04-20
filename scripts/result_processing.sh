#! /bin/bash
ROOT=$(dirname dirname "$0")
cd "$ROOT"
cd "mgnn"

# PDBbind2007
python result_process.py --name PDBbind2007

# PDBbind2013
python result_process.py --name PDBbind2013

# PDBbind2016
python result_process.py --name PDBbind2016

# SARS-CoV-BA K-fold
python result_process.py --name SARS-CoV-BA

# SARS-CoV-BA tested on training PDBbind2016
python result_process.py --name SARS-2016

# PDBbind2016 K-fold
python result_process.py --name kfold-2016

# PDBbind2016 10-orphan
python result_process.py --name orphan-2016