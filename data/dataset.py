import numpy as np
import os
import warnings
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='PDBbind2007', help='PDBbind2007, PDBbind2013, PDBbind2016, PDBbind2019, SARS-CoV-BA')
parser.add_argument('--eta', type=int, default=2)
parser.add_argument('--upsilon', type=int, default=2)
parser.add_argument('--crop_n', type=int, default=56, help='Crop size of sub-graph.')
parser.add_argument('--lorentz', action='store_true', default=False, help='Use Lorentz kernel.')
parser.add_argument('--exponential', action='store_true', default=False, help='Use Exponential kernel.')
parser.add_argument('--graph-path', type=str, default='graph/', help='Path of raw data.')
parser.add_argument('--multi-path', type=str, default='multi/', help='Path for saving parsed data.')
args = parser.parse_args()

name = args.name
eta = args.eta
upsilon = args.upsilon
crop_n = args.crop_n
is_exp = args.exponential
is_lor = args.lorentz
GRAPH_PATH = os.path.abspath(args.graph_path)
MULTI_PATH = os.path.abspath(args.multi_path)

def lorentz(dist, eta):
    return 1/(1+np.power(dist/eta,upsilon))

def exponential(dist, eta):
    return np.exp(-np.power(dist/eta,upsilon))

def save_feature(out_file, adj_matrix_list, feature_list, label_list, id_list, sub_set, save_path=MULTI_PATH):
    fileroot = os.path.join(save_path, out_file)
    if not os.path.exists(fileroot): os.mkdir(fileroot)
    adj_matrix = np.concatenate(adj_matrix_list)
    feature = np.concatenate(feature_list)
    label = np.array(label_list).reshape([-1,1])
    id = np.array(id_list)
    np.save(fileroot+'/adj_matrix_%s'%sub_set, adj_matrix)
    np.save(fileroot+'/feature_%s'%sub_set, feature)
    np.save(fileroot+'/label_%s'%sub_set, label)
    np.save(fileroot+'/id_%s'%sub_set, id)

def with_kernel(dataset_name, sub_set, is_exp=is_exp, crop_n=crop_n, eta=eta, f_case=None):
    if dataset_name.startswith('PDB'): abbr = dataset_name[-4:]
    else: abbr = dataset_name.lower().replace('-','')
    exp = 'exp' if is_exp else 'lor'
    fout = '%s_%s_eta%d_n%d' % (abbr, exp, eta, crop_n)
    print('Generating subgraph for %s %s ...' % (fout, sub_set))
    
    P = np.array(['C', 'N', 'O', 'S'])                              # PROTEIN
    L = np.array(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])   # LIGAND
    root = os.path.join(GRAPH_PATH, dataset_name, 'distance')
    if not f_case: f_case = os.path.join(GRAPH_PATH, dataset_name, '%s.txt' % sub_set)
    f_label = os.path.join(GRAPH_PATH, dataset_name, 'y.txt')
    if sub_set=='core': sub_set='test'
    label_set = {}
    for item in open(f_label).readlines():
        label_set[item.split()[0]] = float(item.split()[1])
    feature_list = []   # [N,36,crop_n,29]
    id_list = []        # [N,]
    label_list = []     # [N,]
    adj_matrix_list = []# [N,36,crop_n,crop_n]

    index=0
    for item in tqdm(open(f_case).readlines()):
        adj_matrix = []
        feature = []
        for i in range(len(P)):
            for j in range(len(L)):
                dist = os.path.join(root,'_'.join([item.split()[0],P[i],L[j],'distance_matrix.csv']))
                feat = os.path.join(root,'_'.join([item.split()[0],P[i],L[j],'feature_matrix.csv']))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    dist = np.loadtxt(dist, delimiter=',')
                if not dist.shape[0]:
                    adj_matrix.append(np.eye(crop_n)[np.newaxis,:,:])
                    feature.append(np.zeros([1,crop_n,29]))
                    continue
                feat = np.loadtxt(feat, delimiter=',')
                if is_lor: dist = lorentz(dist, eta=eta)
                elif is_exp: dist = exponential(dist, eta=eta)
                adj = np.eye(crop_n); feat_pad = np.zeros([crop_n,29])
                sort_idx = np.argsort(sum(dist))[::-1]
                dist = dist[sort_idx]
                dist = dist[:,sort_idx]
                feat = feat[sort_idx]
                cut = min(dist.shape[0], crop_n)
                adj[:cut,:cut]=dist[:cut,:cut]
                feat_pad[:cut,:]=feat[:cut,:]
                adj_matrix.append(adj[np.newaxis,:,:])
                feature.append(feat_pad[np.newaxis,:,:])
        adj_matrix_list.append(np.concatenate(adj_matrix)[np.newaxis,:,:,:])
        feature_list.append(np.concatenate(feature)[np.newaxis,:,:,:])
        id_list.append(item.split()[0])
        label_list.append(label_set[item.split()[0]])
        index += 1
    save_feature(fout, adj_matrix_list, feature_list, label_list, id_list, sub_set)

def no_kernel(dataset_name, sub_set, crop_n=crop_n, f_case=None, suffix=None):
    if dataset_name.startswith('PDB'): abbr = dataset_name[-4:]
    else: abbr = dataset_name.lower().replace('-','')
    fout = '%s_%d' % (abbr, crop_n)
    
    print('Generating subgraph dist & elec for %s %s ...' % (fout, sub_set))
    if suffix: suffix = '_'+suffix
    else: suffix = ''
    warnings.filterwarnings('ignore')
    P = np.array(['C', 'N', 'O', 'S', 'H'])                              # PROTEIN
    L = np.array(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'])   # LIGAND
    root = os.path.join(GRAPH_PATH, dataset_name)
    if not f_case: f_case = os.path.join(root, '%s.txt' % (sub_set))
    f_label = os.path.join(root, 'y.txt')
    label_set = {}
    if sub_set=='core': sub_set='test'
    for item in open(f_label).readlines():
        label_set[item.split()[0]] = float(item.split()[1])

    feature_dist_list = []   # [N,36,crop_n,29]
    feature_charge_list = []
    adj_dist_list = []
    adj_charge_list = []
    label_list = []     # [N,]
    id_list = []
    idx = 0

    for item in tqdm(open(f_case).readlines()):
        item = item.split()[0]
        feature_dist = []
        feature_charge = []
        adj_dist = []
        adj_charge = []
        idx = idx + 1
        for i in range(len(P)):
            for j in range(len(L)):
                feat_d = os.path.join(root, 'distance', '_'.join([item,P[i],L[j],'feature_matrix.csv']))
                feat_c = os.path.join(root, 'charge', '_'.join([item,P[i],L[j],'feature_matrix.csv']))
                adj_d = os.path.join(root, 'distance', '_'.join([item,P[i],L[j],'distance_matrix.csv']))
                adj_c = os.path.join(root, 'charge', '_'.join([item,P[i],L[j],'distance_matrix.csv']))

                if P[i]!='H' and L[j]!='H':
                    feat_d = np.loadtxt(feat_d, delimiter=',')
                    adj_d = np.loadtxt(adj_d, delimiter=',')
                    if not feat_d.shape[0]: 
                        feature_dist.append(np.zeros([1,crop_n,29]))
                        adj_dist.append(np.zeros([1,crop_n,crop_n]))
                    else:
                        cut = min(feat_d.shape[0], crop_n)
                        feat_pad = np.zeros([crop_n, 29])
                        adj_pad = np.zeros([crop_n, crop_n])
                        sort_idx = np.argsort(np.sum(feat_d, axis=1))[::-1]
                        feat_d = feat_d[sort_idx]
                        adj_d = adj_d[sort_idx]
                        feat_pad[:cut, :] = feat_d[:cut, :]
                        adj_pad[:cut,:cut] = adj_d[:cut,:cut]
                        feature_dist.append(feat_pad[np.newaxis,:,:])
                        adj_dist.append(adj_pad[np.newaxis,:,:])
                feat_c = np.loadtxt(feat_c, delimiter=',')
                adj_c = np.loadtxt(adj_c, delimiter=',')
                if not feat_c.shape[0]:
                    feature_charge.append(np.zeros([1, crop_n, 25]))
                    adj_charge.append(np.zeros([1, crop_n, crop_n]))
                else:
                    cut = min(feat_c.shape[0], crop_n)
                    feat_pad = np.zeros([crop_n, 25])
                    adj_pad = np.zeros([crop_n, crop_n])
                    sort_idx = np.argsort(np.sum(feat_c,axis=1))[::-1]
                    feat_c = feat_c[sort_idx][:,:25]
                    adj_c = adj_c[sort_idx]
                    feat_pad[:cut, :] = feat_c[:cut, :]
                    adj_pad[:cut, :cut] = adj_c[:cut, :cut]
                    feature_charge.append(feat_pad[np.newaxis, :, :])
                    adj_charge.append(adj_pad[np.newaxis, :, :])
        feature_dist_list.append(np.concatenate(feature_dist)[np.newaxis,:,:,:])
        feature_charge_list.append(np.concatenate(feature_charge)[np.newaxis,:,:,:])
        adj_dist_list.append(np.concatenate(adj_dist)[np.newaxis,:,:,:])
        adj_charge_list.append(np.concatenate(adj_charge)[np.newaxis,:,:,:])
        id_list.append(item)
        label_list.append(label_set[item])
    save_feature(fout+'_dist'+suffix, adj_dist_list, feature_dist_list, label_list, id_list, sub_set)
    save_feature(fout+'_elec'+suffix, adj_charge_list, feature_charge_list, label_list, id_list, sub_set)

def with_kernel_multisource(name, s_years, save_path, is_exp=is_exp, crop_n=crop_n, eta=eta):
    P = np.array(['C', 'N', 'O', 'S'])                              # PROTEIN
    L = np.array(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])   # LIGAND
    abbr = name[-4:]
    exp = 'exp' if is_exp else 'lor'
    fout = '%s_%s_eta%d_n%d_pretrain' % (abbr, exp, eta, crop_n)

    label_set = {}
    for year in s_years:
        f_label = os.path.join(GRAPH_PATH, 'PDBbind'+str(year), 'y.txt')
        labels = {}
        for item in open(f_label).readlines():
            labels[item.split()[0]] = float(item.split()[1])
        label_set[year] = labels

    feature_list = []   # [N,36,crop_n,29]
    id_list = []        # [N,]
    label_list = []     # [N,]
    adj_matrix_list = []# [N,36,crop_n,crop_n]

    index=0
    print('GCN subgraph input saving to', fout)
    for item in tqdm(open(os.path.join(GRAPH_PATH, name, 'pretrain.txt')).readlines()):
        item = item.strip()
        belong = -1
        for year in s_years:
            if item in label_set[year]:
                belong = year
                break
        root = os.path.join(GRAPH_PATH, 'PDBbind'+str(belong), 'distance')

        adj_matrix = []
        feature = []
        for i in range(len(P)):
            for j in range(len(L)):
                dist = os.path.join(root,'_'.join([item,P[i],L[j],'distance_matrix.csv']))
                feat = os.path.join(root,'_'.join([item,P[i],L[j],'feature_matrix.csv']))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    dist = np.loadtxt(dist, delimiter=',')
                if not dist.shape[0]:
                    adj_matrix.append(np.eye(crop_n)[np.newaxis,:,:])
                    feature.append(np.zeros([1,crop_n,29]))
                    continue
                feat = np.loadtxt(feat, delimiter=',')
                if is_lor: dist = lorentz(dist, eta=eta)
                elif is_exp: dist = exponential(dist, eta=eta)
                adj = np.eye(crop_n); feat_pad = np.zeros([crop_n,29])
                sort_idx = np.argsort(sum(dist))[::-1]
                dist = dist[sort_idx]
                dist = dist[:,sort_idx]
                feat = feat[sort_idx]
                cut = min(dist.shape[0], crop_n)
                adj[:cut,:cut]=dist[:cut,:cut]
                feat_pad[:cut,:]=feat[:cut,:]

                adj_matrix.append(adj[np.newaxis,:,:])
                feature.append(feat_pad[np.newaxis,:,:])
        adj_matrix_list.append(np.concatenate(adj_matrix)[np.newaxis,:,:,:])
        feature_list.append(np.concatenate(feature)[np.newaxis,:,:,:])
        id_list.append(item)
        label_list.append(label_set[belong][item])
        index += 1
    save_feature(fout, adj_matrix_list, feature_list, label_list, id_list, 'pretrain', save_path=save_path)
    print('GCN subgraph input saved to ', fout)
    return label_list

def no_kernel_multisource(name, s_years, save_path, crop_n=crop_n, suffix=None):
    if suffix: suffix = '_'+suffix
    abbr = name[-4:]
    fout = '%s_%d' % (abbr, crop_n)
    warnings.filterwarnings('ignore')
    P = np.array(['C', 'N', 'O', 'S', 'H'])                              # PROTEIN
    L = np.array(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'])   # LIGAND

    label_set = {}
    for year in s_years:
        f_label = os.path.join(GRAPH_PATH,'PDBbind'+str(year),'y.txt')
        labels = {}
        for item in open(f_label).readlines():
            labels[item.split()[0]] = float(item.split()[1])
        label_set[year] = labels

    feature_dist_list = []   # [N,36,crop_n,29]
    feature_charge_list = []
    adj_dist_list = []
    adj_charge_list = []
    label_list = []     # [N,]
    id_list = []

    idx = 0
    print('GCN subgraph input saving to', fout)
    for item in tqdm(open(os.path.join(GRAPH_PATH, name, 'pretrain.txt')).readlines()):
        item = item.strip()
        belong = -1
        for year in s_years:
            if item in label_set[year]:
                belong = year
                break
        root = os.path.join(GRAPH_PATH, 'PDBbind'+str(belong))

        feature_dist = []
        feature_charge = []
        adj_dist = []
        adj_charge = []
        for i in range(len(P)):
            for j in range(len(L)):
                feat_d = os.path.join(root, 'distance', '_'.join([item,P[i],L[j],'feature_matrix.csv']))
                feat_c = os.path.join(root, 'charge', '_'.join([item,P[i],L[j],'feature_matrix.csv']))
                adj_d = os.path.join(root, 'distance', '_'.join([item,P[i],L[j],'distance_matrix.csv']))
                adj_c = os.path.join(root, 'charge', '_'.join([item,P[i],L[j],'distance_matrix.csv']))

                if P[i]!='H' and L[j]!='H':
                    feat_d = np.loadtxt(feat_d, delimiter=',')
                    adj_d = np.loadtxt(adj_d, delimiter=',')
                    if not feat_d.shape[0]:
                        feature_dist.append(np.zeros([1,crop_n,29]))
                        adj_dist.append(np.zeros([1,crop_n,crop_n]))
                    else:
                        cut = min(feat_d.shape[0], crop_n)
                        feat_pad = np.zeros([crop_n, 29])
                        adj_pad = np.zeros([crop_n, crop_n])
                        sort_idx = np.argsort(np.sum(feat_d, axis=1))[::-1]
                        feat_d = feat_d[sort_idx]
                        adj_d = adj_d[sort_idx]
                        feat_pad[:cut, :] = feat_d[:cut, :]
                        adj_pad[:cut,:cut] = adj_d[:cut,:cut]
                        feature_dist.append(feat_pad[np.newaxis,:,:])
                        adj_dist.append(adj_pad[np.newaxis,:,:])
                feat_c = np.loadtxt(feat_c, delimiter=',')
                adj_c = np.loadtxt(adj_c, delimiter=',')
                if not feat_c.shape[0]:
                    feature_charge.append(np.zeros([1, crop_n, 25]))
                    adj_charge.append(np.zeros([1, crop_n, crop_n]))
                else:
                    cut = min(feat_c.shape[0], crop_n)
                    feat_pad = np.zeros([crop_n, 25])
                    adj_pad = np.zeros([crop_n, crop_n])
                    sort_idx = np.argsort(np.sum(feat_c,axis=1))[::-1]
                    feat_c = feat_c[sort_idx][:,:25]
                    adj_c = adj_c[sort_idx]
                    feat_pad[:cut, :] = feat_c[:cut, :]
                    adj_pad[:cut, :cut] = adj_c[:cut, :cut]
                    feature_charge.append(feat_pad[np.newaxis, :, :])
                    adj_charge.append(adj_pad[np.newaxis, :, :])
        idx = idx+1
        feature_dist_list.append(np.concatenate(feature_dist)[np.newaxis,:,:,:])
        feature_charge_list.append(np.concatenate(feature_charge)[np.newaxis,:,:,:])
        adj_dist_list.append(np.concatenate(adj_dist)[np.newaxis,:,:,:])
        adj_charge_list.append(np.concatenate(adj_charge)[np.newaxis,:,:,:])
        id_list.append(item)
        label_list.append(label_set[belong][item])
    save_feature(fout+'_dist'+suffix, adj_dist_list, feature_dist_list, label_list, id_list, 'pretrain', save_path=save_path)
    save_feature(fout+'_elec'+suffix, adj_charge_list, feature_charge_list, label_list, id_list, 'pretrain', save_path=save_path)

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    if is_exp or is_lor:
        if name.startswith('PDB'): with_kernel(name, 'train') # only test_set for SARS-CoV-BA
        if not name.endswith('2019'): with_kernel(name, 'core') # only train_set for PDBbind2019
    else:
        if name.startswith('PDB'): no_kernel(name, 'train')
        if not name.endswith('2019'): no_kernel(name, 'core')
