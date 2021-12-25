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

# SARS-CoV-BA

python result_process.py --name SARS-CoV-BA
