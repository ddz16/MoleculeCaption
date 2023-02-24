import torch
import torch.nn as nn
from transformers import T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from model.gin_model import GNN


class GinDecoder(nn.Module):
    def __init__(self, has_graph=True, MoMuK=True, model_size='base'):
        super(GinDecoder, self).__init__()
        self.has_graph = has_graph
        self.main_model = T5ForConditionalGeneration.from_pretrained("molt5-"+model_size+"-smiles2caption/")
        print(f'hidden_size: {self.main_model.config.hidden_size},\
                d_model: {self.main_model.config.d_model},\
                num_decoder_layers: {self.main_model.config.num_decoder_layers},\
                num_heads: {self.main_model.config.num_heads},\
                d_kv: {self.main_model.config.d_kv}')

        for p in self.main_model.named_parameters():
            p[1].requires_grad = False

        if has_graph:
            self.graph_encoder = GNN(
                num_layer=5,
                emb_dim=300,
                gnn_type='gin',
                drop_ratio=0.0,
                JK='last',
            )
            
            if MoMuK:
                ckpt = torch.load("./MoMu_checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt")
            else:
                ckpt = torch.load("./MoMu_checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt")
            ckpt = ckpt['state_dict']
            pretrained_dict = {k[14:]: v for k, v in ckpt.items()}
            missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(pretrained_dict, strict=False)

            for p in self.graph_encoder.named_parameters():
                p[1].requires_grad = False
            
            self.graph_projector = nn.Sequential(
                nn.Linear(300, self.main_model.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.main_model.config.hidden_size, self.main_model.config.hidden_size)
            )
            # self.graph_projector_dropout = torch.nn.Dropout(0.2)
            # self.graph_projector = nn.Linear(300, self.main_model.config.hidden_size)


    def forward(self, batch, input_ids, encoder_attention_mask, decoder_attention_mask, label):
        device = encoder_attention_mask.device
        B, _ = encoder_attention_mask.shape

        smiles_embeds = self.main_model.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask).last_hidden_state
        
        if self.has_graph:
            graph_rep = self.graph_encoder(batch)
            graph_rep = self.graph_projector(graph_rep)
            smiles_embeds = torch.cat([graph_rep.unsqueeze(1), smiles_embeds], dim=1)
            encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), encoder_attention_mask], dim=1)
        
        encoder_outputs = BaseModelOutput(
                last_hidden_state=smiles_embeds,
                hidden_states=None,
                attentions=None,
            )
        loss = self.main_model(
            encoder_outputs = encoder_outputs,
            attention_mask = encoder_attention_mask,
            decoder_attention_mask = decoder_attention_mask,
            labels=label
            ).loss

        return loss


    def translate(self, batch, input_ids, encoder_attention_mask, tokenizer):
        device = encoder_attention_mask.device
        B, _ = encoder_attention_mask.shape
    
        smiles_embeds = self.main_model.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask).last_hidden_state

        if self.has_graph:
            graph_rep = self.graph_encoder(batch)
            graph_rep = self.graph_projector(graph_rep)
            smiles_embeds = torch.cat([graph_rep.unsqueeze(1), smiles_embeds], dim=1)
            encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), encoder_attention_mask], dim=1)
           
        # input_prompt = ["The molecule is"] * B
        # decoder_input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        # decoder_input_ids = decoder_input_ids.to(device)
        num_beams = 5

        encoder_outputs = BaseModelOutput(
                last_hidden_state=smiles_embeds,
                hidden_states=None,
                attentions=None,
            )
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = encoder_attention_mask,  # important
            num_beams=num_beams,
            max_length=512,
            # eos_token_id=self.main_model.config.eos_token_id,
            # decoder_start_token_id=self.main_model.config.decoder_start_token_id,
            # decoder_input_ids = decoder_input_ids,
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return res

    def gin_encode(self, batch):
        node_reps = self.graph_encoder(batch)
        return node_reps


if __name__ == '__main__':
    model = T5ForConditionalGeneration.from_pretrained("molt5_base/")
    for p in model.named_parameters():
        if 'lm_head' in p[0] or 'shared' in p[0]:
	        print(p[1])
    
    print(model.shared)
    print(model.lm_head)