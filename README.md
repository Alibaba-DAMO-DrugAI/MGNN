Multiphysical Graph Neural Network (MP-GNN)
====
## Abstract
Graph neural networks (GNNs) are the most promising deep learning models that can revolutionize non- Euclidean data analysis. However, their full potential is severely curtailed by poorly represented molecular graphs and features. Here we propose a multiphysical graph neural network (MP-GNN) model based on the developed multiphysical molecular graph representation and featurization. All kinds of molecular interactions, between different atom types and at different scales, are systematically represented by a series of scale-specific and element-specific graphs with distance-related node features. From these graphs, graph convolution network (GCN) models are constructed with specially-designed weight-sharing architectures. Base learners are constructed from GCN models from different elements at different scales, and further consolidated together using both one-scale and multi-scale ensemble learning schemes. Our MP-GNN has two distinct properties. First, our MP-GNN incorporatesmultiscale interactions by using more than one molecular graph. Atomic interactions from various different scales are not modeled by one specific graph (as in traditional GNNs), instead they are represented by a series of graphs at different scales. Second, it is free from the complicated feature generation process as in conventional GNN methods. In our MP-GNN, various atom interactions are embedded into element-specific graph representations with only distance-related node features. A unique GNN architecture is designed to incorporate all the information into a consolidated model. Our MP-GNN has been extensively validated on the widely-used benchmark test datasets from PDBbind, including PDBbind-v2007, PDBbind-v2013, and PDBbind-v2016. Our model can outperform all existing models as far as we know. Further, our MP-GNN is used in COVID-19 drug design. Based on a dataset with 185 complexes of inhibitors for SARS-CoV/SARS-CoV-2, we evaluate their binding affinities using our MP-GNN. It has been found our MP-GNN is of extremely high accuracy. This demonstrates the great potential of our MP-GNN for the screening of potential drugs for SARS-CoV-2.

## Dataset
Download the following dataset to path ```mgnn/data/raw ``` and unzip them. Rename then into PDBbind2007, PDBbind2013, PDBbind2016, and PDBbind2019. 

[1] [PDBbind2007 refined set](http://www.pdbbind.org.cn/download/pdbbind_v2007.tar.gz)

[2] [PDBbind2013 refined set](http://www.pdbbind.org.cn/download/pdbbind_v2013_refined_set.tar.gz)

[3] [PDBbind2016 refined set](http://www.pdbbind.org.cn/download/pdbbind_v2016_refined.tar.gz) 

[4] [PDBbind2019 refined set](http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz)
and [PDBbind2019 general set minus refined set](http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz)

[5] SARS-CoV-BA (already downloaded)

## Requirements
  * [PDB2PQR 2.1.1](https://github.com/Electrostatics/pdb2pqr): calculating the partial charge for atoms in complex. Install to ```mgnn/pdb2pqr ```. 
  * [Open Babel 2.3.90](http://openbabel.org/wiki/Main_Page): transferring the molecule complex into different file formats.
  * PyTorch 1.7
  * Python 3.8

## Data Preprocessing
1. Process the raw data into graph data:
   
   ```bash scripts/parse_raw_data.sh```
2. Process the graph data into 10 resolutions and prepare the pretain datasets:
    
    ```bash scripts/multi_resolution.sh```
3. Process the target-based cross validation split: (for Major Revision)

    ```python data/graph/PDBbind2016/kfold_idx/target_split.py```
4. Process the 10 most frequent orphan-target split: (for Major Revision)

    ```python data/graph/PDBbind2016/orphan_idx/orphan_target.py```
## Training and result processing
All experiments in this section are not assigned to a cuda device. So the computation resource allocation should be down accoding to the hardware condition.

You can achieve this by adding argument ```--gpu=x``` to the bash file.
1. Pretrain for PDBbind2007, PDBbind2013 and PDBbind2016, and save the pretrained models:

    ```bash scripts/pretrain.sh```
2. Finetune for PDBbind2007, PDBbind2013 and PDBbind2016:

    ```bash scripts/finetune.sh```
3. 5-fold cross validation with PDBbind2019 for SARS-Cov-BA:

    ```bash scripts/kfold_sars.sh```
4. Target-based split 5-fold cross validation on PDBbind2016: (for Major Revision)
   
    ```bash scripts/kfold_2016.sh```
5. Orphan-target validation for 10 most frequent protein in PDBbind2016 and for SARS-Cov-BA: (for Major Revision) 
   
    ```bash scripts/orphan_target.sh```
4. For final result processing:

    ```bash scripts/result_processing.sh```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Nguyen D, et al. Unveiling the molecular mechanism of SARS-CoV-2 main protease inhibition from 137 crystal structures using algebraic topology and deep learning[J]. Chemical science, 2020](https://pubs.rsc.org/en/content/articlehtml/2020/sc/d0sc04641h)

## Cite

If you find our work helpful in your research or work, please cite us.

Multiphysical graph neural network (MP-GNN) for COVID-19 drug design at Briefings in Bioinformatics, 2022. （https://doi.org/10.1093/bib/bbac231）

## Questions & Problems
If you have any questions or problems, please feel free to open a new issue. We will fix the new issue ASAP. You can also email the maintainers and authors below.

Xiaoshuang Li, Ying Chi (lixiaoshuang@sjtu.edu.cn, ying.chi@basenbyte.com)

Ying Chi was previously with Alibaba DAMO Academy, directing AI for Drug Discovery team. Now is the co-founder and Chief Technology Officer in Base & Byte Biotechnology Co. Ltd.  
