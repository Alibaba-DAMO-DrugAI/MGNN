import numpy as np
import os
import torch
from utils import load_data, load_train, load_test
import scipy.stats
from math import sqrt
import torch.nn.functional as F
import re
from functools import cmp_to_key
from torch.utils.data import DataLoader
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='PDBbind2007', help='PDBbind2007, PDBbind2013, PDBbind2016, PDBbind2019, SARS-CoV-BA')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%args.gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(ROOT, 'data/graph')
MULTI_PATH = os.path.join(ROOT, 'data/multi')
MODEL_PATH = os.path.join(ROOT, 'best_models')
RESULT_PATH = os.path.join(ROOT, 'results')
LOG_PATH = os.path.join(ROOT, 'log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.terminator = ''
logger.addHandler(sh)

suffixes = ['56_dist', '56_elec', 'exp_eta2_n56', 'exp_eta5_n56', 'exp_eta10_n56', 'exp_eta20_n56',
            'lor_eta2_n56', 'lor_eta5_n56', 'lor_eta10_n56', 'lor_eta20_n56']

class myDataset(torch.utils.data.Dataset):
    def __init__(self, adj, feat, label, idx):
        self.length = adj.shape[0]
        self.dataset = list(zip(adj, feat, label, idx))

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self.length

def compute_coef(a, b, c):
    va, ab, ac = np.cov((a, b, c), bias=True)[0]
    _, vb, bc = np.cov((a, b, c), bias=True)[1]

    m = bc * ab - ac * vb
    n = ac * ab - bc * va
    return m / (m + n), n / (m + n)

def weight(npy_path, y_train):
    pred = [np.load(path) for path in npy_path]
    # pred = npy_path
    pccs = [scipy.stats.pearsonr(p[:len(y_train)], y_train)[0] for p in pred]
    pccs = np.around(pccs, decimals=4)
    mse_train = [F.mse_loss(torch.FloatTensor(pred[i][:len(y_train)]), torch.FloatTensor(y_train))
                 for i in range(0, len(npy_path))]
    rmse_train = np.around(np.sqrt(mse_train), decimals=4)
    
    pcc_w = np.zeros_like(pccs).reshape([1, -1])
    for i in range(0, len(npy_path)): pcc_w[0, i] = pccs[i] / np.sum(pccs)
    rmse_w = np.zeros_like(rmse_train).reshape([1, -1])
    for i in range(0, len(npy_path)): rmse_w[0, i] = rmse_train[i] / np.sum(rmse_train)
    return pcc_w, rmse_w

def joint(weight, npy_path, y_test):
    pcc_w, rmse_w = weight
    pred = [np.load(path) for path in npy_path]
    test_start = len(pred[0]) - len(y_test)
    
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * pcc_w[0, i]
    pcc = scipy.stats.pearsonr(result[test_start:], y_test)[0]
    logger.info('pcc_indicator:%.4f ' % pcc)
    temp = result
    
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * rmse_w[0, i]
    pcc = scipy.stats.pearsonr(result[test_start:], y_test)[0]
    logger.info('rmse_indicator:%.4f\n' % pcc)
    
    return temp

def uniform_joint(pred, y_test):
    test_start = len(pred[0]) - len(y_test)
    mean_w = 0.125 * np.ones([1,10])
    result = np.zeros_like(pred[0])
    for i in range(0, len(pred)): result += pred[i] * mean_w[0, i]
    pcc = scipy.stats.pearsonr(result[test_start:], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[test_start:]), torch.FloatTensor(y_test)))
    tau,_ = scipy.stats.kendalltau(result[test_start:], y_test)
    logger.info('pcc: %.4f, tau:%.4f, rmse:%.4f\n' % (pcc, tau, rmse))
    return pcc

def model_result(model_path, data_path, data=None):
    model = torch.load(model_path)
    if data == None:
        adj_train, dist_train, label_train, adj_test, dist_test, label_test = load_data(data_path)
    else: adj_train, dist_train, label_train, adj_test, dist_test, label_test = data
    train_pred = model(dist_train.cuda(), adj_train.cuda()).cpu().detach().numpy().squeeze()
    test_pred = model(dist_test.cuda(), adj_test.cuda()).cpu().detach().numpy().squeeze()
    result = np.concatenate([train_pred, test_pred], axis=0)
    label = np.concatenate([label_train, label_test], axis=0)
    return result, label


### For PDBbind dataset ###
def gene_npy_pdbbind(name):
    print('Saving predicted results for %s' % name)
    datasets = ['%s_%s' % (name[-4:], suffix) for suffix in suffixes]
    y_train = np.load(os.path.join(GRAPH_PATH, name, 'y_train.npy'))
    y_test = np.load(os.path.join(GRAPH_PATH, name, 'y_test.npy'))
    for i in range(0, 10):
        data = load_data(os.path.join(MULTI_PATH, datasets[i]))
        for seed in range(42, 64):
            save_name = '%s_%d' % (datasets[i], seed)
            model = os.path.join(MODEL_PATH, '%s.models' % save_name)
            pred, label = model_result(model, data_path=None, data=data)
            np.save(os.path.join(RESULT_PATH, save_name), pred)
            pcc_test = scipy.stats.pearsonr(pred[len(y_train):], y_test)[0]
            print('%s test_pcc %.4f result saved.' % (save_name, pcc_test))
    print()

def multi_scale_stacking_pdbbind(name):
    fh = logging.FileHandler(os.path.join(LOG_PATH,'%s_mss.log'%name), mode='w')
    fh.terminator = ''
    logger.addHandler(fh)
    print()
    logger.info('Logging into file %s\n' % os.path.join(LOG_PATH,'%s_mss.log'%name))
    logger.info('PCC: Multi-scale stacking on %s.\n' % name)
    logger.info('6 res (resolutions) means stacking without the Lorentz kernel models.\n\n')
    y_train = np.load(os.path.join(GRAPH_PATH, name, 'y_train.npy'))
    y_test = np.load(os.path.join(GRAPH_PATH, name, 'y_test.npy'))
    for seed in range(42, 64):
        logger.info('Seed_%d, '%seed)
        npy_path = [os.path.join(RESULT_PATH, '%s_%s_%d.npy'%(name[-4:],suffix,seed)) for suffix in suffixes]
        npy_path = np.array(npy_path)

        logger.info('10_res: ')
        w = weight(npy_path, y_train)
        joint(w, npy_path, y_test)

        logger.info('          6_res: ')
        w = weight(npy_path[[0,1,2,3,8,9]], y_train) # without lor dataset
        joint(w, npy_path[[0,1,2,3,8,9]], y_test)
    logger.removeHandler(fh)

def one_scale_stacking_pdbbind(name):
    fh = logging.FileHandler(os.path.join(LOG_PATH, '%s_oss.log' % name), mode='w')
    fh.terminator = ''
    logger.addHandler(fh)
    print()
    logger.info('Logging into file %s\n' % os.path.join(LOG_PATH, '%s_mss.log' % name))
    logger.info('PCC: One-scale stacking on %s.\n\n' % name)

    y_train = np.load(os.path.join(GRAPH_PATH, name, 'y_train.npy'))
    y_test = np.load(os.path.join(GRAPH_PATH, name, 'y_test.npy'))
    final_path = []
    for suffix in suffixes:
        dataset = '%s_%s' % (name[-4:],suffix)
        logger.info(dataset.ljust(19)); logger.info(':')
        npy_path = [os.path.join(RESULT_PATH, '%s_%d.npy'%(dataset,seed)) for seed in range(42,64)]
        npy_path = np.array(npy_path)
        logger.info('')
        w = weight(npy_path, y_train)
        result = joint(w, npy_path, y_test)
        result_filepath = os.path.join(RESULT_PATH, '%s.npy' % dataset)
        np.save(result_filepath, result)
        final_path.append(result_filepath)
    final_path = np.array(final_path)
    logger.info('\nMulti-scale stacking after one-scale stacking:\n')
    logger.info('10_res: ')
    w = weight(final_path, y_train)
    joint(w, final_path, y_test)
    logger.info(' 6_res: ')
    w = weight(final_path[[0, 1, 2, 3, 8, 9]], y_train)
    joint(w, final_path[[0, 1, 2, 3, 8, 9]], y_test)
    logger.removeHandler(fh)


### For PDBbind dataset K-fold cross validation ###
def gene_npy_pdbbind_kfold(name):
    print('Saving predicted results for %s' % name)
    for suffix in suffixes:
        datasets = '%s_%s' % (name[-4:], suffix)
        adj_train, feat_train, label_train, adj_test, feat_test, label_test = load_data(os.path.join(MULTI_PATH, datasets))
        adj_all = torch.cat([adj_train, adj_test], axis=0)
        feat_all = torch.cat([feat_train, feat_test], axis=0)
        label_all = torch.cat([label_train, label_test], axis=0)
        for fold in range(5):
            kfold_test_idx = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s' % name[-4:], 'kfold_idx', '%d.npy' % fold))
            kfold_train_idx = [i for i in range(4056)]
            for i in kfold_test_idx: kfold_train_idx.remove(i)
            adj_train = adj_all[kfold_train_idx]
            feat_train = feat_all[kfold_train_idx]
            label_train = label_all[kfold_train_idx]
            adj_test = adj_all[kfold_test_idx]
            feat_test = feat_all[kfold_test_idx]
            label_test = label_all[kfold_test_idx]
            data = adj_train, feat_train, label_train, adj_test, feat_test, label_test

            save_name = '%s_%d' % (datasets, fold)
            model = os.path.join(MODEL_PATH, '%s.models' % save_name)
            pred, label = model_result(model, data_path=None, data=data)
            np.save(os.path.join(RESULT_PATH, save_name), pred)
            pcc_test = scipy.stats.pearsonr(pred[len(label_train):], label[len(label_train):])[0]
            print('%s test_pcc %.4f result saved.' % (save_name, pcc_test))
        print()

def multi_scale_stacking_pdbbind_kfold(name):
    fh = logging.FileHandler(os.path.join(LOG_PATH,'%s_mss.log'%name), mode='w')
    fh.terminator = ''
    logger.addHandler(fh)
    print()
    logger.info('Logging into file %s\n' % os.path.join(LOG_PATH,'%s_mss.log'%name))
    logger.info('PCC: Multi-scale stacking on %s.\n' % name)
    y_train = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s'%name[-4:], 'y_train.npy'))
    y_test = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s'%name[-4:], 'y_test.npy'))
    y_all = np.concatenate([y_train,y_test], axis=0)
    pcc_fold = []
    for fold in range(5):
        refer_npy = []
        test_npy = []
        train_pcc = []
        test_pcc = []
        kfold_test_idx = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s'%name[-4:], 'kfold_idx', '%d.npy' % fold))
        kfold_train_idx = [i for i in range(4056)]
        for i in kfold_test_idx: kfold_train_idx.remove(i)
        y_train = y_all[kfold_train_idx]
        y_test = y_all[kfold_test_idx]
        logger.info('Fold %d: ' % fold)
        for suffix in suffixes:
            result = np.load(os.path.join(RESULT_PATH, '%s_%s_%d.npy' % (name[-4:], suffix, fold)))
            refer_npy.append(result[:len(y_train)])
            test_npy.append(result[len(y_train):])
            train_pcc.append(scipy.stats.pearsonr(result[:len(y_train)], y_train)[0])
            test_pcc.append(scipy.stats.pearsonr(result[len(y_train):], y_test)[0])
        refer_npy = np.array(refer_npy)
        test_npy = np.array(test_npy)
        logger.info('train: ')
        train_pcc.append(uniform_joint(refer_npy, y_train))
        logger.info('        test: ')
        test_pcc.append(uniform_joint(test_npy, y_test))
        pcc_fold.append(train_pcc)
        pcc_fold.append(test_pcc)
    logger.removeHandler(fh)


### For PDBbind dataset 10 orphan validation ###
def gene_npy_pdbbind_orphan(name):
    print('Saving predicted results for %s' % name)
    for suffix in suffixes:
        datasets = '%s_%s' % (name[-4:], suffix)
        adj_train, feat_train, label_train, adj_test, feat_test, label_test = load_data(os.path.join(MULTI_PATH, datasets))
        adj_all = torch.cat([adj_train, adj_test], axis=0)
        feat_all = torch.cat([feat_train, feat_test], axis=0)
        label_all = torch.cat([label_train, label_test], axis=0)
        for protein in range(10):
            test_idx = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s' % name[-4:], 'orphan_idx', '%d.npy' % protein))
            train_idx = [i for i in range(4056)]
            for i in test_idx: train_idx.remove(i)
            adj_train = adj_all[train_idx]
            feat_train = feat_all[train_idx]
            label_train = label_all[train_idx]
            adj_test = adj_all[test_idx]
            feat_test = feat_all[test_idx]
            label_test = label_all[test_idx]
            data = adj_train, feat_train, label_train, adj_test, feat_test, label_test

            save_name = '%s_p%d' % (datasets, protein)
            model = os.path.join(MODEL_PATH, '%s.models' % save_name)
            pred, label = model_result(model, data_path=None, data=data)
            np.save(os.path.join(RESULT_PATH, save_name), pred)
            pcc_test = scipy.stats.pearsonr(pred[len(label_train):], label[len(label_train):])[0]
            print('%s test_pcc %.4f result saved.' % (save_name, pcc_test))
        print()

def multi_scale_stacking_pdbbind_orphan(name):
    fh = logging.FileHandler(os.path.join(LOG_PATH,'%s_mss.log'%name), mode='w')
    fh.terminator = ''
    logger.addHandler(fh)
    print()
    logger.info('Logging into file %s\n' % os.path.join(LOG_PATH,'%s_mss.log'%name))
    logger.info('PCC: Multi-scale stacking on %s.\n' % name)
    y_train = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s'%name[-4:], 'y_train.npy'))
    y_test = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s'%name[-4:], 'y_test.npy'))
    y_all = np.concatenate([y_train,y_test], axis=0)
    pcc_protein = []
    for protein in range(10):
        refer_npy = []
        test_npy = []
        train_pcc = []
        test_pcc = []
        test_idx = np.load(os.path.join(GRAPH_PATH, 'PDBbind%s'%name[-4:], 'orphan_idx', '%d.npy' % protein))
        train_idx = [i for i in range(4056)]
        for i in test_idx: train_idx.remove(i)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        logger.info('Protein %d: ' % protein)
        for suffix in suffixes:
            result = np.load(os.path.join(RESULT_PATH, '%s_%s_p%d.npy' % (name[-4:], suffix, protein)))
            refer_npy.append(result[:len(y_train)])
            test_npy.append(result[len(y_train):])
            train_pcc.append(scipy.stats.pearsonr(result[:len(y_train)], y_train)[0])
            test_pcc.append(scipy.stats.pearsonr(result[len(y_train):], y_test)[0])
        refer_npy = np.array(refer_npy)
        test_npy = np.array(test_npy)
        logger.info('train: ')
        train_pcc.append(uniform_joint(refer_npy, y_train))
        logger.info('        test: ')
        test_pcc.append(uniform_joint(test_npy, y_test))
        pcc_protein.append(test_pcc)
    logger.removeHandler(fh)


### For k-fold on SARS-CoV-BA dataset ###
def gene_npy_sars_kfold():
    print('Saving predicted results for SARS-CoV-BA.')
    for suffix in suffixes:
        train_path = os.path.join(MULTI_PATH, '2019_%s' % suffix)
        test_path = os.path.join(MULTI_PATH, 'sarscovba_%s' % suffix)
        adj_2019, feat_2019, label_2019 = load_train(train_path)
        adj_sars, feat_sars, label_sars = load_test(test_path)
        for fold in range(5):
            kfold_test_idx = np.load(os.path.join(GRAPH_PATH, 'SARS-CoV-BA/kfold_idx', '%d.npy' % fold))
            kfold_train_idx = [i for i in range(185)]
            for i in kfold_test_idx: kfold_train_idx.remove(i)
            
            adj_train = torch.cat([adj_2019, adj_sars[kfold_train_idx]], dim=0)
            feat_train = torch.cat([feat_2019, feat_sars[kfold_train_idx]], dim=0)
            label_train = torch.cat([label_2019, label_sars[kfold_train_idx]], dim=0)
            adj_test = adj_sars[kfold_test_idx]
            feat_test = feat_sars[kfold_test_idx]
            label_test = label_sars[kfold_test_idx]
            
            idx_train = np.arange(adj_train.shape[0])
            train_set = myDataset(adj_train, feat_train, label_train, idx_train)
            train_loader = DataLoader(train_set, batch_size=6000, shuffle=False, drop_last=False)
            idx_test = np.arange(adj_test.shape[0])
            test_set = myDataset(adj_test, feat_test, label_test, idx_test)
            test_loader = DataLoader(test_set, batch_size=6000, shuffle=False, drop_last=False)
            
            model = torch.load(os.path.join(MODEL_PATH, '%s_%d.models' % (suffix, fold)))
            model = model.cuda()
            model.eval()
            result_train = []
            for data in train_loader:
                adj, feat, label, idx = data
                output = model(feat.cuda(), adj.cuda()).cpu().detach().numpy().reshape(-1)
                result_train.append(output)
            for data in test_loader:
                adj, feat, label, _ = data
                result_test = model(feat.cuda(), adj.cuda()).cpu().detach().numpy().reshape(-1)
            result_train = np.concatenate(result_train, axis=0).reshape(-1)
            result_test = np.array(result_test).reshape(-1)
            result = np.concatenate([result_train, result_test], axis=0)
            
            np.save(os.path.join(RESULT_PATH, '%s_%d.npy'%(suffix,fold)), result)
            pcc_test = scipy.stats.pearsonr(result_test, np.array(label_test).reshape(-1))[0]
            print('%s_%d test_pcc %.4f result saved.' % (suffix, fold, pcc_test))
        print()

def multi_scale_stacking_sars_kfold():
    fh = logging.FileHandler(os.path.join(LOG_PATH,'sarscovba_mss.log'), mode='w')
    fh.terminator = ''
    logger.addHandler(fh)
    print()
    logger.info('Logging into file %s\n' % os.path.join(LOG_PATH,'sarscovba_mss.log'))
    logger.info('PCC: Multi-scale stacking on SARS-CoV-BA.\n')
    
    for fold in range(5):
        refer_npy = []
        test_npy = []
        logger.info('Fold %d: ' % fold)
        for suffix in suffixes:
            path_2019 = os.path.join(MULTI_PATH, '2019_%s'% (suffix))
            path_sars = os.path.join(MULTI_PATH, 'sarscovba_%s'% (suffix))
            y_2019 = np.load(path_2019 + '/label_train.npy')
            y_sars = np.load(path_sars + '/label_test.npy')
            
            kfold_test_idx = np.load(os.path.join(GRAPH_PATH, 'SARS-CoV-BA/kfold_idx', '%d.npy' % fold))
            kfold_train_idx = [i for i in range(185)]
            for i in kfold_test_idx: kfold_train_idx.remove(i)
            
            y_train = np.concatenate([y_2019, y_sars[kfold_train_idx]], axis=0).reshape(-1)
            y_test = y_sars[kfold_test_idx].reshape(-1)
            
            result = np.load(os.path.join(RESULT_PATH,'%s_%d.npy'%(suffix,fold)))
            refer_npy.append(result[:len(y_train)])
            test_npy.append(result[len(y_train):])
        refer_npy = np.array(refer_npy)
        test_npy = np.array(test_npy)
        logger.info('train: ')
        uniform_joint(refer_npy, y_train)
        logger.info('        test: ')
        uniform_joint(test_npy, y_test)
    logger.removeHandler(fh)


### For training on PDBbind20xx and testing on SARS-CoV-BA dataset ###
def gene_npy_sars_20xx(name):
    print('Saving predicted results for SARS-CoV-BA.')
    for suffix in suffixes:
        train_path = os.path.join(MULTI_PATH, '%s_%s' % (name[-4:], suffix))
        test_path = os.path.join(MULTI_PATH, 'sarscovba_%s' % suffix)
        adj_train, feat_train, label_train, adj_test, feat_test, label_test = load_data(train_path)
        adj_train = torch.cat([adj_train, adj_test],axis=0)
        feat_train = torch.cat([feat_train, feat_test], axis=0)
        label_train = torch.cat([label_train, label_test], axis=0)

        adj_test, feat_test, label_test = load_test(test_path)
        data = adj_train, feat_train, label_train, adj_test, feat_test, label_test

        model = os.path.join(MODEL_PATH, 'sarscovba_%s.models' % (suffix))
        pred, label = model_result(model, data_path=None, data=data)
        np.save(os.path.join(RESULT_PATH, 'sarscovba_%s.npy' % (suffix)), pred)
        pcc_test = scipy.stats.pearsonr(pred[len(label_train):], np.array(label_test).reshape(-1))[0]
        print('sarscovba_%s test_pcc %.4f result saved.' % (suffix, pcc_test))

def multi_scale_stacking_sars_20xx(name):
    fh = logging.FileHandler(os.path.join(LOG_PATH, 'sarscovba_%s_mss.log'%(name[-4:])), mode='w')
    fh.terminator = ''
    logger.addHandler(fh)
    print()
    logger.info('Logging into file %s\n' % os.path.join(LOG_PATH, 'sarscovba_%s_mss.log'%(name[-4:])))
    logger.info('PCC: Multi-scale stacking on SARS-CoV-BA.\n')

    refer_npy = []
    test_npy = []
    test_pcc = []
    for suffix in suffixes:
        path_train = os.path.join(MULTI_PATH, '%s_%s' % (name[-4:],suffix))
        path_sars = os.path.join(MULTI_PATH, 'sarscovba_%s' % (suffix))
        y_train = np.concatenate([np.load(path_train + '/label_train.npy'), np.load(path_train + '/label_test.npy')],axis=0)
        y_test = np.load(path_sars + '/label_test.npy')

        result = np.load(os.path.join(RESULT_PATH, 'sarscovba_%s.npy' % (suffix)))
        print(result.shape, y_test.shape)
        refer_npy.append(result[:len(y_train)])
        test_npy.append(result[len(y_train):])
        test_pcc.append(scipy.stats.pearsonr(result[len(y_train):], y_test)[0])
    refer_npy = np.array(refer_npy)
    test_npy = np.array(test_npy)
    logger.info('train: ')
    uniform_joint(refer_npy, y_train)
    logger.info('        test: ')
    test_pcc.append(uniform_joint(test_npy, y_test))
    logger.removeHandler(fh)

    for item in test_pcc:
        print('%.4f' % item, end=' ')
    print()


### For Summary ###
def pdbbind20xx(name):
    gene_npy_pdbbind(name)
    multi_scale_stacking_pdbbind(name)
    one_scale_stacking_pdbbind(name)

def kfold_sars():
    gene_npy_sars_kfold()
    multi_scale_stacking_sars_kfold()

def sars_20xx(name):
    gene_npy_sars_20xx(name)
    multi_scale_stacking_sars_20xx(name)

def kfold_20xx(name):
    gene_npy_pdbbind_kfold(name)
    multi_scale_stacking_pdbbind_kfold(name)

def orphan(name):
    gene_npy_pdbbind_orphan(name)
    multi_scale_stacking_pdbbind_orphan(name)

if __name__ == '__main__':
    if args.name.startswith('PDB'): pdbbind20xx(args.name)
    elif args.name=='SARS-CoV-BA': kfold_sars()
    elif args.name.startswith('SARS'): sars_20xx(args.name)
    elif args.name.startswith('orphan'): orphan(args.name)
    else: kfold_20xx(args.name)