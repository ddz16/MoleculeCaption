
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

import argparse
import sys

from model.GinT5 import GinDecoder
# from model.GinGPT import GinDecoder, ClipCaptionModel, ClipCaptionPrefix
from model.gin_model import GNN
from dataloader import TextMoleculeReplaceDataset
from tqdm import tqdm

#import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='mode')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size')
parser.add_argument('--nlayers', type=int, default=6, help='number of layers')
parser.add_argument('--emb_size', type=int, default=512, help='input dimension size')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--max_smiles_length', type=int, default=512, help='max smiles length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--nhead', type=int, default=8, help='num attention heads')

parser.add_argument('--MoMuK', default=False, action='store_true')
parser.add_argument('--model_size', type=str, default='base')
parser.add_argument('--data_path', type=str, default='data/', help='path where data is located =')
parser.add_argument('--saved_path', type=str, default='saved_models/', help='path where weights are saved')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

parser.add_argument('--use_scheduler', type=bool, default=True, help='Use linear scheduler')
parser.add_argument('--num_warmup_steps', type=int, default=400, help='Warmup steps for linear scheduler, if enabled.')

parser.add_argument('--output_file', type=str, default='out.txt', help='path where test generations are saved')

args = parser.parse_args()

runseed = 100
torch.manual_seed(runseed)
np.random.seed(runseed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(runseed)


tokenizer = T5Tokenizer.from_pretrained("molt5-"+args.model_size+"-smiles2caption/", model_max_length=512)

train_data = TextMoleculeReplaceDataset(args.data_path, 'train', tokenizer)
val_data = TextMoleculeReplaceDataset(args.data_path, 'validation', tokenizer)
test_data = TextMoleculeReplaceDataset(args.data_path, 'test', tokenizer)

train_dataloader = torch_geometric.loader.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)#, collate_fn=pad_collate)
val_dataloader = torch_geometric.loader.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)#, collate_fn=pad_collate)
test_dataloader = torch_geometric.loader.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)#, collate_fn=pad_collate)

if args.MoMuK:
    print("model init with MoMu-K")
else:
    print("model init with MoMu-S")

my_model = GinDecoder(has_graph=True, MoMuK=args.MoMuK, model_size=args.model_size).to(device)

if args.mode == 'test':
    state_dict = torch.load('saved_models/gint5_smiles2caption_'+args.model_size+'.pt')
    my_model.load_state_dict(state_dict)

if args.mode == 'train':
    for p in my_model.named_parameters():
    	if p[1].requires_grad:
            print(p[0])

    pg = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr)
    # num_training_steps = args.epochs * len(train_dataloader) - args.num_warmup_steps
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps, num_training_steps = num_training_steps) 

MAX_LENGTH = args.max_length


def train_epoch(dataloader, model, optimizer, epoch):
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    
    # model.train()
    losses = 0
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # model.zero_grad()
    
        graph = d['graph'].to(device)

        smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
        smiles_tokens = smiles_tokens_['input_ids'].to(device)
        src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
        
        text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
        text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
        label = text_tokens_['input_ids'].to(device)  # caption

        label = label.masked_fill(~text_mask.bool(), -100)

        # print(smiles_tokens)
        # print(src_padding_mask)
        # print(label)
        # print(text_mask)

        loss = model(graph, smiles_tokens, src_padding_mask, text_mask, label)

        if j % 300 == 0: 
            print('total steps: {}, step: {}, loss: {}'.format(epoch*len(dataloader) + j, j, loss))

        loss.backward()
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()

        losses += loss.item()

    return losses / len(dataloader)


def eval(dataloader, model, epoch):
    model.eval()
    losses = 0
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):

            graph = d['graph'].to(device)

            smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
            
            text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
            text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            label = text_tokens_['input_ids'].to(device)  # caption

            label = label.masked_fill(~text_mask.bool(), -100)

            loss = model(graph, smiles_tokens, src_padding_mask, text_mask, label)
            losses += loss.item()
            if j % 100 == 0:
                print('val total steps: {}, step: {}, val loss: {}'.format(epoch*len(dataloader) + j, j, loss))
    
    return losses/len(dataloader)


if args.mode == 'train':
    # my_model.train()
    min_val_loss = 10000
    for i in range(args.epochs):
        print('Epoch:', i)
        train_epoch(train_dataloader, model=my_model, optimizer=optimizer, epoch=i)
        val_loss = eval(val_dataloader, model=my_model, epoch=i)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("--------------------save model--------------------")
            torch.save(my_model.state_dict(), args.saved_path + 'gint5_smiles2caption_' + args.model_size + '.pt')


if args.mode == 'test':
    my_model.eval()
    smiles = []
    test_outputs = []
    test_gt = []
    with torch.no_grad():
        for j, d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            real_text = d['description']
            graph = d['graph'].to(device)

            smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
            
            # text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
            # text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            # label = text_tokens_['input_ids'].to(device)  # caption
          
            outputs = my_model.translate(graph, smiles_tokens, src_padding_mask, tokenizer)

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