import numpy as np
import dataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='PDBbind2007', help='PDBbind2007, PDBbind2013, PDBbind2016, PDBbind2019, SARS-CoV-BA')
parser.add_argument('--eta', type=int, default=20)
parser.add_argument('--crop_n', type=int, default=56, help='Crop size of sub-graph.')
parser.add_argument('--lorentz', action='store_true', default=False, help='Use Lorentz kernel.')
parser.add_argument('--exponential', action='store_true', default=True, help='Use Exponential kernel.')
parser.add_argument('--graph-path', type=str, default='graph/', help='Path of raw data.')
parser.add_argument('--pretrain-path', type=str, default='pretrain/', help='Path for saving parsed data.')
args = parser.parse_args()

name = args.name
eta = args.eta
crop_n = args.crop_n
is_exp = args.exponential
is_lor = args.lorentz
GRAPH_PATH = os.path.abspath(args.graph_path)
PRETRAIN_PATH = os.path.abspath(args.pretrain_path)
years = [2007, 2013, 2016]
year_set = set(years)

def gene_pretrain_multisource(name):
    year = int(name[-4:])
    ref_years = year_set - {year}

    dataset.with_kernel_multisource(name, ref_years, PRETRAIN_PATH, is_exp=is_exp, crop_n=crop_n, eta=eta)
    dataset.no_kernel_multisource(name, ref_years, PRETRAIN_PATH, crop_n=crop_n, suffix='pretrain')

if __name__=='__main__':
    gene_pretrain_multisource(name)