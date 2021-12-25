import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_pretrain
from model import GCN
import os
import scipy.stats as sp
import matplotlib.pyplot as plt
import logging
import shutil

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='2016_29_exp_eta5_n100')
parser.add_argument('--outfile', type=str, default='test')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'pretrain_models')
LOG_PATH = os.path.join(ROOT, 'log/%s'%(args.outfile))
if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
shutil.copy('pretrain.py', LOG_PATH)
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
DATA_PATH = os.path.join(ROOT, 'data/pretrain', args.dataset)
adj, feat, label = load_pretrain(DATA_PATH)

# Model and optimizer
model = GCN(nfeat=feat.shape[-1], nhid=hidden, dropout=args.dropout, nsub=feat.shape[1])
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    feat = feat.cuda()
    adj = adj.cuda()
    label = label.cuda()

def train(epoch, max_pcc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(feat, adj)
    mse = F.mse_loss(output, label)
    output = output.cpu().detach().numpy().reshape(-1)
    l = label.cpu().detach().numpy().reshape(-1)
    pcc = sp.pearsonr(output, l)
    mse.backward()
    optimizer.step()

    lr = str(optimizer.param_groups[0]['lr'])
    if pcc[0] > max_pcc:
        max_pcc = pcc[0]
        update = False
        for model_file in os.listdir(MODEL_PATH):
            if model_file.startswith(args.dataset):
                update = True
                os.remove(os.path.join(MODEL_PATH,model_file))
                torch.save(model, os.path.join(MODEL_PATH,'%s.models'%args.dataset))
        if not update: torch.save(model, os.path.join(MODEL_PATH,'%s.models'%args.dataset))
        logger.info('Epoch: {:04d} '.format(epoch+1) +
              'lr: {} '.format(lr) +
              'mse: {:.4f} '.format(mse.item()) +
              'pcc: {:.4f} '.format(pcc[0]) +
              'time: {:.4f}s '.format(time.time() - t))
    return max_pcc

# Train model
t_total = time.time()
max_pcc = 0.0
for epoch in range(args.epochs):
    if epoch in epoch_index:
        optimizer.param_groups[0]['lr'] *= lr_decay
    max_pcc = train(epoch, max_pcc)
logger.info("Optimization Finished!")
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
