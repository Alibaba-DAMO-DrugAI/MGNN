import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import scipy.stats as sp
import matplotlib.pyplot as plt
import logging

from utils import load_train, load_test
from model import GCN
import shutil
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2016)
parser.add_argument('--dataset', type=str, default='exp_eta2_n100')
parser.add_argument('--outfile', type=str, default='test')
parser.add_argument('--fold', type=int, default=0, help='K-fold index number.')
parser.add_argument('--batch_size', type=int, default=6000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3400,help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'best_models')
LOG_PATH = os.path.join(ROOT, 'log', 'sarscovba_'+args.dataset)
if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
shutil.copy('sars.py', LOG_PATH)
shutil.copy('model.py', LOG_PATH)
shutil.copy('layers.py', LOG_PATH)

epoch_index=[800,1600,2400,3200,4000,4800,5600]
lr_decay=0.5
log_name = os.path.join(LOG_PATH, 'log.txt')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_name, mode='w')
sh = logging.StreamHandler()
logger.addHandler(fh); logger.addHandler(sh)
logger.info('Logging into file %s' % LOG_PATH)
logger.info(str(args))
logger.info('Decay index: '+str(epoch_index))
hidden = [64,16,16,256]
logger.info('hidden: '+str(hidden))

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

save_name = 'sarscovba_'+args.dataset
class myDataset(torch.utils.data.Dataset):
    def __init__(self, adj, feat, label, idx):
        self.length = idx.shape[0]
        self.dataset = list(zip(adj, feat, label, idx))
    def __getitem__(self, item):
        return self.dataset[item]
    def __len__(self):
        return self.length

# Load data
TRAIN_PATH = os.path.join(ROOT, 'data/multi', '%d_%s'%(args.year, args.dataset)) # 4056
TEST_PATH = os.path.join(ROOT, 'data/multi', 'sarscovba_%s'%args.dataset) # 185
adj_train, feat_train, label_train = load_train(TRAIN_PATH)
adj_test, feat_test, label_test = load_test(TEST_PATH)

# Model and optimizer
model = GCN(nfeat=feat_train.shape[-1],nhid=hidden,dropout=args.dropout,nsub=feat_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    feat_train = feat_train.cuda()
    adj_train = adj_train.cuda()
    label_train = label_train.cuda()
    feat_test = feat_test.cuda()
    adj_test = adj_test.cuda()
    label_test = label_test.cuda()
    
idx_train = np.arange(adj_train.shape[0])
train_set = myDataset(adj_train, feat_train, label_train, idx_train)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
idx_test = np.arange(adj_test.shape[0])
test_set = myDataset(adj_test, feat_test, label_test, idx_test)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

def train(epoch, max_pcc, train_loader, test_loader):
    total_mse = 0
    batch_num = 0
    output = []
    label = []
    for data in train_loader:
        model.train()
        adj_train, feat_train, label_train, idx_train = data
        optimizer.zero_grad()
        output_train = model(feat_train, adj_train)
        mse_train = F.mse_loss(output_train, label_train)
        output_train = output_train.cpu().detach().numpy().reshape(-1)
        label_train = label_train.cpu().detach().numpy().reshape(-1)
        mse_train.backward()
        optimizer.step()
        
        total_mse += mse_train.to('cpu').data.numpy()
        output.append(output_train)
        label.append(label_train)
        batch_num = batch_num + 1
    output_train = np.concatenate(output, axis=0)
    label_train = np.concatenate(label, axis=0)
    pcc_train = sp.pearsonr(output_train, label_train)[0]

    total_mse = 0
    batch_num = 0
    output = []
    label = []
    for data in test_loader:
        model.eval()
        adj_test, feat_test, label_test, _ = data
        output_test = model(feat_test, adj_test)
        mse_test = F.mse_loss(output_test, label_test)
        output_test = output_test.cpu().detach().numpy().reshape(-1)
        label_test = label_test.cpu().detach().numpy().reshape(-1)

        total_mse += mse_test.to('cpu').data.numpy()
        output.append(output_test)
        label.append(label_test)
        batch_num = batch_num + 1
    mse_test = total_mse/batch_num
    output_test = np.concatenate(output, axis=0)
    label_test = np.concatenate(label, axis=0)
    pcc_test = sp.pearsonr(output_test, label_test)[0]

    for data in train_loader:
        model.eval()
        adj_eval, feat_eval, label_eval, idx_eval = data
        output_eval = model(feat_eval, adj_eval)
        output_eval = output_eval.cpu().detach().numpy().reshape(-1)
        label_eval = label_eval.cpu().detach().numpy().reshape(-1)
        output.append(output_eval)
        label.append(label_eval)
    output_eval = np.concatenate(output, axis=0)
    label_eval = np.concatenate(label, axis=0)
    pcc_eval = sp.pearsonr(output_eval, label_eval)[0]
    
    if pcc_test > max_pcc:
        max_pcc = pcc_test
        update = False
        for model_file in os.listdir(MODEL_PATH):
            if model_file.startswith(save_name):
                update = True
                os.remove(os.path.join(MODEL_PATH, model_file))
                torch.save(model, os.path.join(MODEL_PATH, '%s.models' % save_name))
        if not update: torch.save(model, os.path.join(MODEL_PATH, '%s.models' % save_name))
        logger.info('Epoch: {:04d} '.format(epoch + 1) +
                    'lr: {} '.format(optimizer.param_groups[0]['lr']) +
                    'pcc_train: {:.4f} '.format(pcc_train) +
                    '/ {:.4f} '.format(pcc_eval) +
                    'mse_val: {:.4f} '.format(mse_test) +
                    'pcc_val: {:.4f} '.format(pcc_test) +
                    'time: {:.4f}s '.format(time.time() - t))
    return max_pcc

# Train model
t_total = time.time()
max_pcc = 0.0
decay_count = 0
stop_count = 0
t = time.time()
for epoch in range(args.epochs):
    if stop_count > 1300: break
    if decay_count > 500:
        optimizer.param_groups[0]['lr'] *= lr_decay
        decay_count = 0
    last_max = max_pcc
    max_pcc = train(epoch, max_pcc, train_loader, test_loader)
    if max_pcc > last_max:
        decay_count = 0
        stop_count = 0
    else: 
        decay_count += 1
        stop_count += 1
    
logger.info("Optimization Finished for %s!"%save_name)
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
