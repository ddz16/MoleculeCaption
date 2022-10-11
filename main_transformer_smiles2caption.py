
#Based on https://pytorch.org/tutorials/beginner/translation_transformer.html

from lib2to3.pgen2 import token
import torch_geometric
import torch
from transformers import AutoTokenizer, LogitsProcessorList, BeamSearchScorer, BertTokenizer, T5Tokenizer
from torch import nn
import torch.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from torch.utils.data.dataloader import default_collate

from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np

import pickle
from model.GinGPT import MappingType
import argparse
import sys

from model.GinT5 import GinDecoder
# from model.GinGPT import GinDecoder, ClipCaptionModel, ClipCaptionPrefix
from model.gin_model import GNN
from dataloader import TextMoleculeReplaceDataset
from models_baseline import Seq2SeqTransformer
#import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='mode')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size')
parser.add_argument('--nlayers', type=int, default=6, help='number of layers')
parser.add_argument('--emb_size', type=int, default=512, help='input dimension size')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--max_smiles_length', type=int, default=512, help='max smiles length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--nhead', type=int, default=8, help='num attention heads')

parser.add_argument('--MoMuK', type=bool, default=False, action='store_true')
parser.add_argument('--model_size', type=str, default='base')
parser.add_argument('--data_path', type=str, default='data/', help='path where data is located =')
parser.add_argument('--saved_path', type=str, default='saved_models/', help='path where weights are saved')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

parser.add_argument('--use_scheduler', type=bool, default=True, help='Use linear scheduler')
parser.add_argument('--num_warmup_steps', type=int, default=400, help='Warmup steps for linear scheduler, if enabled.')

parser.add_argument('--output_file', type=str, default='out.txt', help='path where test generations are saved')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("molt5-"+args.model_size+"/", model_max_length=512)  # BertTokenizer.from_pretrained('scibert/')

train_data = TextMoleculeReplaceDataset(args.data_path, 'train', tokenizer)
val_data = TextMoleculeReplaceDataset(args.data_path, 'validation', tokenizer)
test_data = TextMoleculeReplaceDataset(args.data_path, 'test', tokenizer)


def build_smiles_vocab(dicts):
    smiles = []
    for d in dicts:
        for cid in d:
            smiles.append(d[cid])

    char_set = set()

    for smi in smiles:
        for c in smi:
            char_set.add(c)

    return ''.join(char_set)

class SmilesTokenizer():

    def __init__(self, smiles_vocab, max_len=512):
        self.smiles_vocab = smiles_vocab
        self.max_len = max_len
        self.vocab_size = len(smiles_vocab) + 3 #SOS, EOS, pad
        
        self.SOS = self.vocab_size - 2
        self.EOS = self.vocab_size - 1
        self.pad = 0

    def letterToIndex(self, letter):
        return self.smiles_vocab.find(letter) + 1 #skip 0 == [PAD]
        
    def ind2Letter(self, ind):
        if ind == self.SOS: return '[SOS]'
        if ind == self.EOS: return '[EOS]'
        if ind == self.pad: return '[PAD]'
        return self.smiles_vocab[ind-1]
        
    def decode(self, iter):
        return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')

    def __len__(self):
        return self.vocab_size

    def get_tensor(self, smi):
        tensor = torch.zeros(1, args.max_smiles_length, dtype=torch.int64)
        tensor[0,0] = smiles_tokenizer.SOS
        for li, letter in enumerate(smi):
            tensor[0,li+1] = self.letterToIndex(letter)
            if li + 3 == args.max_smiles_length: break
        tensor[0, li+2] = self.EOS

        return tensor

smiles_vocab = build_smiles_vocab((train_data.cids_to_smiles, val_data.cids_to_smiles, test_data.cids_to_smiles))
smiles_tokenizer = SmilesTokenizer(smiles_vocab)

train_data.smiles_tokenizer = smiles_tokenizer
val_data.smiles_tokenizer = smiles_tokenizer
test_data.smiles_tokenizer = smiles_tokenizer


train_dataloader = torch_geometric.loader.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)#, collate_fn=pad_collate)
val_dataloader = torch_geometric.loader.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)#, collate_fn=pad_collate)
test_dataloader = torch_geometric.loader.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)#, collate_fn=pad_collate)


model = GinDecoder(has_graph=True, MoMuK=args.MoMuK, model_size=args.model_size).to(device)

if args.mode == 'test':
    state_dict = torch.load('saved_models/gint5_smiles2caption_epoch40.pt')
    model.load_state_dict(state_dict)

if args.mode == 'train':
    for p in model.named_parameters():
    	if p[1].requires_grad:
            print(p[0])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr)
    # num_training_steps = args.epochs * len(train_dataloader) - args.num_warmup_steps
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps, num_training_steps = num_training_steps) 

PAD_IDX = 0 #note that both vocabularies share the same padding token

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

MAX_LENGTH = args.max_length


def train_epoch(dataloader, model, optimizer, epoch):
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    
    model.train()
    losses = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for j, d in enumerate(dataloader):
        model.zero_grad()
        if j % 1000 == 0: print('Step:', j)

        graph = d['graph'].to(device)

        smiles_tokens = d['smiles_tokens'].to(device)

        src_padding_mask = d['smiles_mask'].to(device)  # encoder输入的mask

        text_mask = d['text_mask'].to(device)  # caption的mask，即decoder输入的mask

        label = d['text'].to(device)  # caption本身
        label = label.masked_fill(~text_mask.bool(), -100)

        loss = model(graph, smiles_tokens, src_padding_mask, text_mask, label)

        loss.backward()
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()
        print('total steps: {}, step: {}, loss: {}'.format(epoch*len(dataloader) + j, j, loss))
        losses += loss.item()

    return losses / len(dataloader)


def eval(dataloader, model, epoch):
    model.eval()
    losses = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for j, d in enumerate(dataloader):
            if j % 100 == 0: print('Val Step:', j)

            graph = d['graph'].to(device)

            smiles_tokens = d['smiles_tokens'].to(device)

            src_padding_mask = d['smiles_mask'].to(device)  # encoder输入的mask

            text_mask = d['text_mask'].to(device)  # caption的mask，即decoder输入的mask

            label = d['text'].to(device)  # caption本身
            label = label.masked_fill(~text_mask.bool(), -100)

            loss = model(graph, smiles_tokens, src_padding_mask, text_mask, label)
            losses += loss.item()
            print('val total steps: {}, step: {}, val loss: {}'.format(epoch*len(dataloader) + j, j, loss))

    return losses/len(dataloader)


if args.mode == 'train':
    min_val_loss = 10000
    for i in range(args.epochs):
        print('Epoch:', i)
        train_epoch(train_dataloader, model, optimizer, i)
        val_loss = eval(val_dataloader, model, i)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("save model")
            torch.save(model.state_dict(), args.saved_path + 'gint5_smiles2caption_epoch' + str(i) + '.pt')


if args.mode == 'test':
    model.eval()
    smiles = []
    test_outputs = []
    test_gt = []
    with torch.no_grad():
        for j, d in enumerate(test_dataloader):
            real_text = d['description']
            graph = d['graph'].to(device)

            smiles_tokens = d['smiles_tokens'].to(device)

            src_padding_mask = d['smiles_mask'].to(device)  # encoder输入的mask

            text_mask = d['text_mask'].to(device)  # caption的mask，即decoder输入的mask
          
            outputs = model.translate(graph, smiles_tokens, src_padding_mask, tokenizer)

            # print(outputs)
            # break

            smiles.extend(d['smiles'])
            test_gt.extend(real_text)
            test_outputs.extend(outputs)
            
    with open(args.output_file, 'w') as f:
        f.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
        for smi, rt, ot in zip(smiles, test_gt, test_outputs):
            f.write(smi + '\t' + rt + '\t' + ot + '\n')

# def test_eval(dataloader, model):
#     model.eval()
#     smiles = []
#     test_outputs = []
#     test_gt = []
#     with torch.no_grad():
#         for j, d in enumerate(dataloader):
#             if j % 100 == 0: print('Test Step:', j)
#             graph = d['graph'].to(device)
#             labels = d['text'].to(device)
#             labels[labels == tokenizer.pad_token_id] = -100
#             print(model.translate(graph, labels, tokenizer))
#             # real_text = d['description']
#             # smiles.extend(d['smiles'])
#             # test_gt.extend(real_text)
            
#             # test_outputs.extend([graph2caption(model, graph, tokenizer) for smi in d['smiles']])

#             #wandb.log({'test total steps':len(dataloader) + j, 'step':j,'test loss' : loss})

#     return smiles, test_gt, test_outputs

# smiles, test_gt, test_outputs = test_eval(test_dataloader, model)


#wandb.finish()