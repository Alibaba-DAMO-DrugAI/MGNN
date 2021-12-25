import numpy as np
import scipy.sparse as sp
import torch

epsilon = 1e-10

def encode_onehot(labels):
    labels = labels.reshape(-1)
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def preprocessing(features, adjs, labels):
    num_cases, num_subgraph, num_nodes, _ = adjs.shape
    for i in range(num_cases):
        adj=adjs[i]
        for j in range(num_subgraph):
            sub = adj[j]
            sub = sub + np.multiply(sub.T, sub.T > sub) - np.multiply(sub, sub.T > sub)
            sums = np.sum(sub,axis=-1)
            sub = np.multiply(sub, np.power(sums.astype(float)+epsilon,-1))
            adj[j,:,:] = sub
        adjs[i,:,:,:]=adj
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    adjs = torch.FloatTensor(adjs)

    return features, adjs, labels

def load_data(path):
    feat_train = np.load(path+'/feature_train.npy')
    adj_train = np.load(path+'/adj_matrix_train.npy')
    label_train = np.load(path+'/label_train.npy')
    feat_train, adj_train, label_train = preprocessing(feat_train, adj_train, label_train)

    feat_test = np.load(path+'/feature_test.npy')
    adj_test = np.load(path+'/adj_matrix_test.npy')
    label_test = np.load(path+'/label_test.npy')
    feat_test, adj_test, label_test = preprocessing(feat_test, adj_test, label_test)

    return adj_train, feat_train, label_train, adj_test, feat_test, label_test

def load_pretrain(path):
    feat_pretrain = np.load(path + '/feature_pretrain.npy')
    adj_pretrain = np.load(path + '/adj_matrix_pretrain.npy')
    label_pretrain = np.load(path + '/label_pretrain.npy')
    feat_pretrain, adj_pretrain, label_pretrain = preprocessing(feat_pretrain, adj_pretrain, label_pretrain)
    return adj_pretrain, feat_pretrain, label_pretrain

def load_train(path):
    feat = np.load(path + '/feature_train.npy')
    adj = np.load(path + '/adj_matrix_train.npy')
    label = np.load(path + '/label_train.npy')
    feat, adj, label = preprocessing(feat, adj, label)
    return adj, feat, label

def load_test(path):
    feat = np.load(path + '/feature_test.npy')
    adj = np.load(path + '/adj_matrix_test.npy')
    label = np.load(path + '/label_test.npy')
    feat, adj, label = preprocessing(feat, adj, label)
    return adj, feat, label
