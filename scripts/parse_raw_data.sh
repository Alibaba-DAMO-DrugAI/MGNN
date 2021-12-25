#! /bin/bash
ROOT=$(dirname dirname "$0")
cd "$ROOT"
cd "data"

python rawdata_parser.py --name='PDBbind2007'

python rawdata_parser.py --name='PDBbind2013'

python rawdata_parser.py --name='PDBbind2016'

python rawdata_parser.py --name='PDBbind2019'

python rawdata_parser.py --name='SARS-CoV-BA'