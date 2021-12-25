import os
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='PDBbind2007',
                    help='PDBbind2007, PDBbind2013, PDBbind2016 or PDBbind2019')
parser.add_argument('--raw-path', type=str, default='raw/', help='Path of raw data.')
parser.add_argument('--save-path', type=str, default='graph/', help='Path for saving parsed data.')
parser.add_argument('--pdb2pqr-path', type=str, default='../pdb2pqr/', help='Path for pdb2pqr library.')
args = parser.parse_args()

SAVE_ROOT = os.path.abspath(args.save_path)
RAW_ROOT = os.path.abspath(args.raw_path)
epsilon = 1e-8

def read_mol2(file):
    f = open(file)
    switch = False
    atoms = []
    for line in f.readlines():
      if line[0]=='@':
        if line.split('>')[1].strip()=='ATOM':
            switch = True
        else: switch = False
      elif switch:
        atom = line.split()[2:5]
        atom.append(line.split()[5].split('.')[0]) # atom type
        atom.append(line.split()[-1]) # charge
        atoms.append(atom)
    atoms = np.array(atoms, dtype=str)
    return atoms

def read_pdb(file):
    f = open(file)
    atoms = []
    for line in f.readlines():
      if line.split()[0]=='ATOM':
        atom = [line[30:38]]
        atom.append(line[38:46])
        atom.append(line[46:54])
        atom.append(line[13]) # atom type
        atoms.append(atom)
    atoms = np.array(atoms, dtype=str)
    return atoms

def read_pqr(file):
    f = open(file)
    atoms = []
    for line in f.readlines():
      if line.split()[0]=='ATOM':
        atom = [line[30:38].strip()]
        atom.append(line[38:46].strip())
        atom.append(line[46:54].strip())
        atom.append(line[12:16].strip()[0]) # atom type
        atom.append(line[55:62].strip()) # charge
        atoms.append(atom)
    atoms = np.array(atoms, dtype=str)
    return atoms

def read_sdf(file):
    f = open(file)
    atoms = []
    for line in f.readlines():
        if len(line.split())==10 and line.split()[3] in L:
            atom = [line.split()[0]]
            atom.append(line.split()[1])
            atom.append(line.split()[2])
            atom.append(line.split()[3])  # atom type
            atoms.append(atom)
    atoms = np.array(atoms, dtype=str)
    return atoms

def pdb2pqr(name, ids):
    print('Generating PRQ for %s...'%name)
    wd = args.pdb2pqr_path
    os.chdir(wd)
    root = RAW_ROOT + name
    for id in tqdm(ids):
        pdb = '%s/%s/%s_protein.pdb' % (root, id, id)
        pqr = '%s/%s/%s_protein.pqr' % (root, id, id)
        os.system('python pdb2pqr.py --ff=parse --ph-calc-method=propka --with-ph=7.0 %s %s' % (pdb, pqr))

def pdb2pqr_sars():
    print('Generating PRQ for %s...' % name)
    wd = args.pdb2pqr_path
    os.chdir(wd)
    root = RAW_ROOT + '/SARS-CoV-BA'
    for id in os.listdir(root):
        sdf = os.path.join(root, id, '%s_ligand.pdb' % id)
        mol2 = os.path.join(root, id, '%s_ligand.mol2' % id)
        pdb = os.path.join(root, id, '%s_pocket.pdb' % id)
        pqr = os.path.join(root, id, '%s_pocket.pqr' % id)
        os.system('python pdb2pqr.py --ff=parse --ph-calc-method=propka --with-ph=7.0 %s %s' % (pdb, pqr))
        os.system('obabel %s %s' % (sdf, mol2))
        
def gene_distance_pdb(name, ids):
    P = np.array(['C', 'N', 'O', 'S'])
    L = np.array(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    raw_root = RAW_ROOT + name
    matrix_root = SAVE_ROOT + name + '/distance'
    if not os.path.exists(matrix_root): os.mkdir(matrix_root)
    for id in tqdm(ids):
      pocket = '%s/%s/%s_pocket.pdb' % (raw_root, id, id)
      ligand = '%s/%s/%s_ligand.mol2' % (raw_root, id, id)
      pocket = read_pdb(pocket)
      ligand = read_mol2(ligand)
      for p in P:
        for l in L:
            p_idx = np.argwhere(pocket[:,3]==p)
            l_idx = np.argwhere(ligand[:,3]==l)
            if not os.path.exists('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l)):
                print('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l))
            if not os.path.exists('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l)):
                print('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l))
            if p_idx.shape[0]==0 or l_idx.shape[0]==0:
              open('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l), 'w')
              open('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l), 'w')
            else:
              p_co = pocket[:,0:3][p_idx.squeeze()].reshape([-1,3])
              p_co = np.expand_dims(np.array(p_co, dtype=float),axis=1)
              l_co = ligand[:,0:3][l_idx.squeeze()].reshape([-1,3])
              l_co = np.expand_dims(np.array(l_co, dtype=float),axis=0)
              p_num = p_co.shape[0]; l_num = l_co.shape[1]
              p_co = np.repeat(p_co, l_num, axis=1)
              l_co = np.repeat(l_co, p_num, axis=0)
              dist = np.sqrt(np.sum(np.power((p_co-l_co),2),axis=2))
              dist_mat = 999*(np.ones([p_num+l_num, p_num+l_num])-np.eye(p_num+l_num))
              dist_mat[:p_num,p_num:] = dist
              dist_mat[p_num:,:p_num] = dist.T
              np.savetxt('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l), dist_mat, delimiter=',',fmt='%.01f')

              dist_mat[dist_mat==0] = 999
              nest = np.append(np.arange(2+1,30+1,step=1,dtype=float),100)
              mask = np.zeros_like(dist_mat, dtype=int)
              freq = np.zeros([p_num+l_num, 30])
              for i in nest: mask += dist_mat>i
              for i in range(mask.shape[0]):
                for j in range(mask.shape[1]): freq[i,mask[i,j]]+=1
              np.savetxt('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l),freq[:,:freq.shape[1]-1], delimiter=',', fmt='%d')

def gene_distance_sars():
    P = np.array(['C', 'N', 'O', 'S'])
    L = np.array(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    raw_root = RAW_ROOT + '/SARS-CoV-BA'
    matrix_root = SAVE_ROOT + '/SARS-CoV-BA' + '/distance'
    if not os.path.exists(matrix_root): os.mkdir(matrix_root)
    for id in tqdm(os.listdir(raw_root)):
      pocket = '%s/%s/%s_pocket.pdb' % (raw_root, id, id)
      ligand = '%s/%s/%s_ligand.mol2' % (raw_root, id, id)
      pocket = read_pdb(pocket)
      ligand = read_mol2(ligand)
      for p in P:
        for l in L:
            p_idx = np.argwhere(pocket[:,3]==p)
            l_idx = np.argwhere(ligand[:,3]==l)
            if p_idx.shape[0]==0 or l_idx.shape[0]==0:
              open('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l), 'w')
              open('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l), 'w')
            else:
              p_co = pocket[:,0:3][p_idx.squeeze()].reshape([-1,3])
              p_co = np.expand_dims(np.array(p_co, dtype=float),axis=1)
              l_co = ligand[:,0:3][l_idx.squeeze()].reshape([-1,3])
              l_co = np.expand_dims(np.array(l_co, dtype=float),axis=0)
              p_num = p_co.shape[0]; l_num = l_co.shape[1]
              p_co = np.repeat(p_co, l_num, axis=1)
              l_co = np.repeat(l_co, p_num, axis=0)
              dist = np.sqrt(np.sum(np.power((p_co-l_co),2),axis=2))
              dist_mat = 999*(np.ones([p_num+l_num, p_num+l_num])-np.eye(p_num+l_num))
              dist_mat[:p_num,p_num:] = dist
              dist_mat[p_num:,:p_num] = dist.T
              np.savetxt('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l), dist_mat, delimiter=',',fmt='%.01f')

              dist_mat[dist_mat==0] = 999
              nest = np.append(np.arange(2+1,30+1,step=1,dtype=float),100)
              mask = np.zeros_like(dist_mat, dtype=int)
              freq = np.zeros([p_num+l_num, 30])
              for i in nest: mask += dist_mat>i
              for i in range(mask.shape[0]):
                for j in range(mask.shape[1]): freq[i,mask[i,j]]+=1
              np.savetxt('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l),freq[:,:freq.shape[1]-1], delimiter=',', fmt='%d')

def gene_charge_pdb(name, ids):
    P = np.array(['C', 'N', 'O', 'S', 'H'])
    L = np.array(['C', 'N', 'O', 'S', 'H', 'P', 'F', 'Cl', 'Br', 'I'])
    raw_root = RAW_ROOT + name
    matrix_root = SAVE_ROOT + name + '/charge'
    if not os.path.exists(matrix_root): os.mkdir(matrix_root)
    nest = np.arange(0+0.04, 1+0.04, step=0.04, dtype=float)
    for id in tqdm(ids):
      # id = 'review-com11'
      pocket = '%s/%s/%s_pocket.pqr' % (raw_root, id, id)
      ligand = '%s/%s/%s_ligand.mol2' % (raw_root, id, id)
      pocket = read_pqr(pocket)
      ligand = read_mol2(ligand)
      for p in P:
        for l in L:
            distance_file = '%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l)
            feature_file = '%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l)
            p_idx = np.argwhere(pocket[:,3]==p)
            l_idx = np.argwhere(ligand[:,3]==l)
            # if p=='C' and l=='S': print(p_idx.shape, l_idx.shape)
            if p_idx.shape[0]==0 or l_idx.shape[0]==0:
              open(distance_file, 'w')
              open(feature_file, 'w')
            else:
              p_co = pocket[:,0:3][p_idx.squeeze()].reshape([-1,3])
              p_co = np.expand_dims(np.array(p_co, dtype=float),axis=1)
              p_charge = pocket[:,4][p_idx.squeeze()].reshape([-1])
              p_charge = np.expand_dims(np.array(p_charge, dtype=float), axis=1)
              l_co = ligand[:,0:3][l_idx.squeeze()].reshape([-1,3])
              l_co = np.expand_dims(np.array(l_co, dtype=float),axis=0)
              l_charge = ligand[:, 4][l_idx.squeeze()].reshape([-1])
              l_charge = np.expand_dims(np.array(l_charge, dtype=float), axis=0)
              p_num = p_co.shape[0]; l_num = l_co.shape[1]
              p_co = np.repeat(p_co, l_num, axis=1)
              l_co = np.repeat(l_co, p_num, axis=0)
              p_charge = np.repeat(p_charge, l_num, axis=1)
              l_charge = np.repeat(l_charge, p_num, axis=0)
              dist = np.sqrt(np.sum(np.power((p_co-l_co),2),axis=2))
              dist = 1/(1+np.exp(-100*p_charge*l_charge/dist))
              dist_mat = 999*(np.ones([p_num+l_num, p_num+l_num])-np.eye(p_num+l_num))
              dist_mat[:p_num,p_num:] = dist
              dist_mat[p_num:,:p_num] = dist.T
              np.savetxt(distance_file, dist_mat, delimiter=',',fmt='%.03f')

              dist_mat[dist_mat==0] = 999
              mask = np.zeros_like(dist_mat, dtype=int)
              freq = np.zeros([p_num+l_num, 26])
              for i in nest: mask += dist_mat>i
              for i in range(mask.shape[0]):
                for j in range(mask.shape[1]): freq[i,mask[i,j]]+=1
              np.savetxt(feature_file,freq[:,:freq.shape[1]-1], delimiter=',', fmt='%d')

def gene_charge_sars():
    P = np.array(['C', 'N', 'O', 'S', 'H'])
    L = np.array(['C', 'N', 'O', 'S', 'H', 'P', 'F', 'Cl', 'Br', 'I'])
    raw_root = RAW_ROOT + '/SARS-CoV-BA'
    matrix_root = SAVE_ROOT + '/SARS-CoV-BA' + '/charge'
    if not os.path.exists(matrix_root): os.mkdir(matrix_root)
    nest = np.arange(0+0.04, 1+0.04, step=0.04, dtype=float)
    for id in tqdm(os.listdir(raw_root)):
      pocket = '%s/%s/%s_pocket.pqr' % (raw_root, id, id)
      ligand = '%s/%s/%s_ligand.mol2' % (raw_root, id, id)
      pocket = read_pqr(pocket)
      ligand = read_mol2(ligand)
      for p in P:
        for l in L:
            p_idx = np.argwhere(pocket[:,3]==p)
            l_idx = np.argwhere(ligand[:,3]==l)
            if p_idx.shape[0]==0 or l_idx.shape[0]==0:
              open('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l), 'w')
              open('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l), 'w')
            else:
              p_co = pocket[:,0:3][p_idx.squeeze()].reshape([-1,3])
              p_co = np.expand_dims(np.array(p_co, dtype=float),axis=1)
              p_charge = pocket[:,4][p_idx.squeeze()].reshape([-1])
              p_charge = np.expand_dims(np.array(p_charge, dtype=float), axis=1)
              l_co = ligand[:,0:3][l_idx.squeeze()].reshape([-1,3])
              l_co = np.expand_dims(np.array(l_co, dtype=float),axis=0)
              l_charge = ligand[:, 4][l_idx.squeeze()].reshape([-1])
              l_charge = np.expand_dims(np.array(l_charge, dtype=float), axis=0)
              p_num = p_co.shape[0]; l_num = l_co.shape[1]
              p_co = np.repeat(p_co, l_num, axis=1)
              l_co = np.repeat(l_co, p_num, axis=0)
              p_charge = np.repeat(p_charge, l_num, axis=1)
              l_charge = np.repeat(l_charge, p_num, axis=0)
              dist = np.sqrt(np.sum(np.power((p_co-l_co),2),axis=2))
              dist = np.where(dist==0, 0.5, dist)
              dist = 1/(1+np.exp(-100*p_charge*l_charge/dist))
              dist_mat = 999*(np.ones([p_num+l_num, p_num+l_num])-np.eye(p_num+l_num))
              dist_mat[:p_num,p_num:] = dist
              dist_mat[p_num:,:p_num] = dist.T
              np.savetxt('%s/%s_%s_%s_distance_matrix.csv' % (matrix_root, id, p, l), dist_mat, delimiter=',',fmt='%.03f')

              dist_mat[dist_mat==0] = 999
              mask = np.zeros_like(dist_mat, dtype=int)
              freq = np.zeros([p_num+l_num, 26])
              for i in nest: mask += dist_mat>i
              for i in range(mask.shape[0]):
                for j in range(mask.shape[1]): freq[i,mask[i,j]]+=1
              np.savetxt('%s/%s_%s_%s_feature_matrix.csv' % (matrix_root, id, p, l),freq[:,:freq.shape[1]-1], delimiter=',', fmt='%d')

if __name__ == '__main__':
    name = args.name
    label = open(SAVE_ROOT+'/%s/y.txt' % (name), 'r')
    label_set = {}
    for line in label.read().splitlines():
        label_set[line.split()[0]] = line.split()[1]
        
    if name.startswith('PDB'):
        train_id = open(SAVE_ROOT+'/%s/train.txt' % (name), 'r')
        train_id = train_id.read().splitlines()
        train_id = list(set(train_id))
        test_id = open(SAVE_ROOT+'/%s/core.txt' % (name), 'r')
        test_id = test_id.read().splitlines()
        ids = train_id + test_id
        ids.sort()

        pdb2pqr(name, ids)
        gene_distance_pdb(name, ids)
        gene_charge_pdb(name, ids)
    else:
        # pdb2pqr_sars() # already been down
        gene_distance_sars()
        gene_charge_sars()