from ogb.utils import smiles2graph
import torch_geometric
from torch_geometric.data import Data, Dataset
import torch
from transformers import T5Tokenizer
import os.path as osp

import csv
import pickle

import spacy


class TextMoleculeReplaceDataset(Dataset): #This dataset replaces the name of the molecule at the beginning of the description
    def __init__(self, data_path, split, tokenizer):
        self.data_path = data_path
        
        self.tokenizer = tokenizer

        self.cids = []
        self.descriptions = {}
    
        self.cids_to_smiles = {}
        self.smiles = {}
        
        #load data
       
        with open(osp.join(data_path, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for n, line in enumerate(reader):
                self.descriptions[line['CID']] = line['description']
                self.cids_to_smiles[line['CID']] = line['SMILES']
                self.cids.append(line['CID'])
        
        # if split == 'train':
        #     self.cids = [1,2,3]
        #     self.cids_to_smiles = {
        #         1: 'CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO[C@@H]1[C@@H]([C@H]([C@H]([C@H](O1)CO)OCC2=CC=C(C=C2)F)O)O)[C@@H]([C@@H](CCCCCCCCCCCCCC)O)O',
        #         2: 'CC(=O)N[C@@H]1[C@H](C[C@](O[C@H]1[C@@H]([C@@H](CO)O)O)(C(=O)O)OC[C@@H]2[C@@H]([C@@H]([C@H]([C@H](O2)O)NC(=O)C)O[C@H]3[C@@H]([C@H]([C@H]([C@H](O3)CO)O)O)O)O)O',
        #         3: 'CCCCCCCCCCCCC1=CC=C(C=C1)S(=O)(=O)O',
        #         }
        #     self.descriptions = {
        #         1: 'The molecule is a glycophytoceramide having a 4-O-(4-fluorobenzyl)-alpha-D-galactosyl residue at the O-1 position and a hexacosanoyl group attached to the nitrogen. One of a series of an extensive set of 4"-O-alkylated alpha-GalCer analogues evaluated (PMID:30556652) as invariant natural killer T-cell (iNKT) antigens. It derives from an alpha-D-galactose.',
        #         2: 'The molecule is a branched amino trisaccharide that consists of N-acetyl-alpha-D-galactosamine having a beta-D-galactosyl residue attached at the 3-position and a beta-N-acetylneuraminosyl residue attached at the 6-position. It has a role as an epitope. It is an amino trisaccharide and a galactosamine oligosaccharide.',
        #         3: 'The molecule is a member of the class dodecylbenzenesulfonic acids that is benzenesulfonic acid in which the hydrogen at position 4 of the phenyl ring is substituted by a dodecyl group.',
        #     }


    def __len__(self):
        return len(self.cids)


    def __getitem__(self, idx):

        cid = self.cids[idx]

        smiles = self.cids_to_smiles[cid]

        description = self.descriptions[cid]

        ori_graph = smiles2graph(smiles)
        x = torch.from_numpy(ori_graph['node_feat']).to(torch.int64)
        # print(x)
        edge_index = torch.from_numpy(ori_graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(ori_graph['edge_feat']).to(torch.int64)
        num_nodes = int(ori_graph['num_nodes'])
        graph = Data(x, edge_index, edge_attr, num_nodes=num_nodes)

        # text = self.tokenizer(description, return_tensors='pt')

        # smiles_tokens = self.tokenizer(smiles, return_tensors='pt')

        return {'graph': graph, 'smiles':smiles, 'description':description}


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("molt5_base/", model_max_length=512)
    mydataset = TextMoleculeReplaceDataset('data/', 'test', tokenizer)
    train_loader = torch_geometric.loader.DataLoader(
            mydataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            # persistent_workers = True
        )
    for i, a in enumerate(train_loader):
        print(a)
        text_mask = a['text_mask']
        label = a['text']  # caption
        label = label.masked_fill(~text_mask.bool(), -100)
        print(label)
        break
