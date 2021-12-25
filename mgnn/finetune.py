import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data
from model import GCN
import os
import scipy.stats as sp
import matplotlib.pyplot as plt
import logging
import shutil


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='2016_29_exp_eta5_n100')
parser.add_argument('--pretrain', type=str, default=None)
parser.add_argument('--outfile', type=str, default='test')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=6400, help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-5,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'best_models')
LOG_PATH = os.path.join(ROOT, 'log', args.dataset+'_'+str(args.seed))
if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
shutil.copy('finetune.py', LOG_PATH)
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

# Load data
DATA_PATH = os.path.join(ROOT, 'data/multi', args.dataset)
adj_train, feat_train, label_train, adj_test, feat_test, label_test = load_data(DATA_PATH)

# Model and optimizer
if args.pretrain:
    pretrain_model = os.path.join(ROOT, 'pretrain_models', args.pretrain)
    model = torch.load(pretrain_model)
else:
    model = GCN(nfeat=feat_train.shape[-1],nhid=hidden,dropout=args.dropout,nsub=feat_train.shape[1])

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
save_name = args.dataset+'_'+str(args.seed)

if args.cuda:
    model.cuda()
    feat_train = feat_train.cuda()
    adj_train = adj_train.cuda()
    label_train = label_train.cuda()
    feat_test = feat_test.cuda()
    adj_test = adj_test.cuda()
    label_test = label_test.cuda()

def train(epoch, max_pcc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output_train = model(feat_train, adj_train)
    mse_train = F.mse_loss(output_train, label_train)
    output_train = output_train.cpu().detach().numpy().reshape(-1)
    l_train = label_train.cpu().detach().numpy().reshape(-1)
    pcc_train = sp.pearsonr(output_train, l_train)
    mse_train.backward()
    optimizer.step()

    model.eval()
    output_test = model(feat_test, adj_test)
    mse_test = F.mse_loss(output_test, label_test)
    output_test = output_test.cpu().detach().numpy().reshape(-1)
    l_test = label_test.cpu().detach().numpy().reshape(-1)
    pcc_test = sp.pearsonr(output_test, l_test)
    lr = str(optimizer.param_groups[0]['lr'])
    if pcc_test[0] > max_pcc:
        max_pcc = pcc_test[0]
        update = False
        for model_file in os.listdir(MODEL_PATH):
            if model_file.startswith(save_name):
                update = True
                os.remove(os.path.join(MODEL_PATH,model_file))
                torch.save(model, os.path.join(MODEL_PATH,'%s.models'%save_name))
        if not update: torch.save(model, os.path.join(MODEL_PATH,'%s.models'%save_name))
        logger.info('Epoch: {:04d} '.format(epoch+1) +
              'lr: {} '.format(lr) +
              'mse_train: {:.4f} '.format(mse_train.item()) +
              'pcc_train: {:.4f} '.format(pcc_train[0]) +
              'mse_test: {:.4f} '.format(mse_test.item()) +
              'pcc_test: {:.4f} '.format(pcc_test[0]) +
              'time: {:.4f}s '.format(time.time() - t))
    return max_pcc, lr, pcc_train[0], pcc_test[0]

# Train model
t_total = time.time()
max_pcc = 0.0
for epoch in range(args.epochs):
    if epoch in epoch_index:
        optimizer.param_groups[0]['lr'] *= lr_decay
    max_pcc,lr,pcc_train,pcc_test = train(epoch, max_pcc)
logger.info("Optimization Finished!")
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

