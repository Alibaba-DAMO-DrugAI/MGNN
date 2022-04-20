import os
import numpy as np
import heapq

def gene_protein_id():
    fin = open('index/INDEX_refined_name.2016').readlines()
    name_dict = dict()
    p_idx = 0
    for line in fin:
        if line[0]=='#': continue
        protein = line[20:].strip()
        if not protein in name_dict:
            name_dict[protein] = p_idx
            p_idx += 1
    np.save('protein_id.npy', name_dict)
    print('Prontein ID dictionary saved to protein_id.npy.')
    idx_dict = dict()
    p_idx = 0
    max_len = 0
    for name in name_dict:
        idx_dict[p_idx] = name
        p_idx += 1
        if max_len<len(name): max_len = len(name)
    fout = open('protein_id.txt','w')
    for i in range(len(idx_dict)):
        fout.write('%-51s %d\n' %(idx_dict[i], i))
    fout.close()
    print('Prontein ID saved to protein_id.txt.')

def complex_protein_id():
    fin = open('index/INDEX_refined_name.2016').readlines()
    protein_id_dict = np.load('protein_id.npy',allow_pickle=True).item()
    complex_protein_id = dict()
    for line in fin:
        if line[0]=='#': continue
        complex = line[:4].strip()
        protein = line[20:].strip()
        complex_protein_id[complex] = protein_id_dict[protein]
    # Mean complex number in each protein type
    count = np.zeros(len(protein_id_dict))
    for key in complex_protein_id:
        count[complex_protein_id[key]] += 1
    max_idx = heapq.nlargest(10, range(len(count)), count.take)

    # Align with the used set
    data = np.array([line.split()[0] for line in open('../y.txt').readlines()])
    all_complex = [complex for complex in complex_protein_id]
    for complex in all_complex:
        if not complex in data: complex_protein_id.pop(complex)
    np.save('complex_protein_id.npy', complex_protein_id)
    fout = open('complex_protein_id.txt', 'w')
    for i in complex_protein_id:
        fout.write('%s %d\n' % (i, complex_protein_id[i]))
    fout.close()

    for i,idx in enumerate(max_idx):
        family = [k for k,v in complex_protein_id.items() if v==idx]
        for k in protein_id_dict:
            if protein_id_dict[k]==idx: protein=k
        print('Protein %d: id=%d, size=%d, name=%s'%(i, idx, len(family), protein))
        idx_list = []
        for item in family: idx_list.append(np.where(data==item)[0][0])
        idx_list.sort()
        np.save('%d.npy'%i,idx_list)

if __name__=='__main__':
    gene_protein_id()
    complex_protein_id()