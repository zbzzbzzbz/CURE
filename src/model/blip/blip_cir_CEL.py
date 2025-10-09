from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.blip.med import BertModel
from src.tools.utils import print_dist


class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        med_config="configs/med_config.json",
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
        tar_mm_loss_weight=0.4,
        CEL_loss_weight=0.2,
        le_loss_weight=1.0,
        lkl_loss_weight=0.01,
        lral_loss_weight=1.0,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        dropout = 0.5
        self.combiner_fc = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim),
                                         nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(embed_dim, 1),
                                       nn.Sigmoid())

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False


        self.temp = 0.015
        self.CEL_loss_weight = CEL_loss_weight
        self.lkl_loss_weight = lkl_loss_weight
        self.le_loss_weight = le_loss_weight
        self.lral_loss_weight = lral_loss_weight
        # The scaling parameter of evidence extractor
        self.tau = 0.1
        print(f"self.temp : {self.temp} | self.tau : {self.tau}")
        print(f"self.CEL_loss_weight : {self.CEL_loss_weight} | self.lkl_loss_weight : {self.lkl_loss_weight} | self.le_loss_weight : {self.le_loss_weight} | self.lral_loss_weight : {self.lral_loss_weight}")
        
        self.tar_mm_loss_weight = tar_mm_loss_weight
        print(f"tar_mm_loss_weight: {tar_mm_loss_weight}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"self.device : {self.device}")


    def compute_query_embs(self, batch):
        caption = batch["edit"]
        device = self.device
        if self.train_vit:
            ref_img_embs = self.visual_encoder(batch["ref_img"])
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(batch["ref_img"])

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        query_si_embs = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_si_feat = query_si_embs.last_hidden_state[:, 0, :]
        query_si_feat = F.normalize(self.text_proj(query_si_feat), dim=-1)
        
        # second multi-modal feature computation goes here 
        description = batch["description"]
        mod_text_raw = [desc + ", but" + cap for desc, cap in zip(description, caption)]

        mod_text = self.tokenizer(
            mod_text_raw,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids2 = mod_text.input_ids.clone()
        
        encoder_input_ids2[:, 0] = self.tokenizer.enc_token_id
        query_embs2 = self.text_encoder(
            encoder_input_ids2,
            attention_mask=mod_text.attention_mask,
            mode="text",
            return_dict=True,
        )
        query_feat2 = query_embs2.last_hidden_state[:, 0, :]
        query_feat2 = F.normalize(self.text_proj(query_feat2), dim=-1)
        
        combined_feature = self.combiner_fc(torch.cat([query_si_feat, query_feat2], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query_si_feat = dynamic_scaler * query_si_feat + (1 - dynamic_scaler) * query_feat2
        
        # query_si_feat = query_si_feat*0.8 + query_feat2*0.2

        return query_si_feat

    def compute_mm_loss(self, batch, query_si_feat, preds, uncertainty):
        tar_mmfeat = batch["target_mmemb"].to(self.device)
        tar_img_mmfeat = F.normalize(tar_mmfeat, dim=-1)

        mm_loss = self.loss(query_si_feat, tar_img_mmfeat, self.temp, preds, uncertainty)
        return mm_loss
        


    def forward(self, batch, fabric, preds, uncertainty):
        tar_img_feat = batch["tar_img_feat"]
        device = self.device

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = F.normalize(tar_img_feat, dim=-1)
        # Compute the query image
        query_si_feat = self.compute_query_embs(batch)

        # CEL
        batch_length = query_si_feat.size(0)
        alpha_i2t, alpha_t2i, _, sims_tanh, _ = self.get_alpha(query_si_feat, tar_img_feat)
        batch_labels = torch.eye(batch_length)
        n_idx = (1 - preds).nonzero().view(1, -1)[0].tolist()
        c_idx = preds.nonzero().view(1, -1)[0].tolist()
        for i in n_idx:
            batch_labels[i][i] = 0
        batch_labels = batch_labels.to(device).long()
        loss_edl = mse_loss(batch_labels, alpha_i2t, batch_length, self.lkl_loss_weight, self.le_loss_weight)
        loss_edl += mse_loss(batch_labels, alpha_t2i, batch_length, self.lkl_loss_weight, self.le_loss_weight)
        loss_edl = torch.mean(loss_edl)

        loss = 0
        si_ti_loss = self.loss(query_si_feat, tar_img_feat, self.temp, c_idx, uncertainty)
        loss += self.lral_loss_weight * si_ti_loss
        
        if "target_mmemb" in batch:
            loss += self.tar_mm_loss_weight * self.compute_mm_loss(batch, query_si_feat, c_idx, uncertainty)
        
        final_loss = self.CEL_loss_weight * loss_edl + loss

        return final_loss

    
    def warmup_batch(self, batch, fabric):
        tar_img_feat = batch["tar_img_feat"]
        device = self.device

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = F.normalize(tar_img_feat, dim=-1)
        # Compute the query image
        query_si_feat = self.compute_query_embs(batch)

        # CEL
        alpha_i2t, alpha_t2i, _, sims_tanh, _ = self.get_alpha(query_si_feat, tar_img_feat)

        batch_length = query_si_feat.size(0)
        batch_labels = torch.eye(batch_length).to(device).long()

        loss_edl = mse_loss(batch_labels, alpha_i2t, batch_length, self.lkl_loss_weight, self.le_loss_weight)
        loss_edl += mse_loss(batch_labels, alpha_t2i, batch_length, self.lkl_loss_weight, self.le_loss_weight)
        loss_edl = torch.mean(loss_edl) 

        labels_allone = torch.ones(batch_length).bool().to(device)

        loss = 0
        si_ti_loss = self.loss(query_si_feat, tar_img_feat, self.temp, labels_allone, None)
        loss += self.lral_loss_weight * si_ti_loss

        if "target_mmemb" in batch:
            loss += self.tar_mm_loss_weight * self.compute_mm_loss(batch, query_si_feat, labels_allone, None)
            
        final_loss = self.CEL_loss_weight * loss_edl + loss

        return final_loss


    def get_alpha(self, query_si_feat, tar_img_feat):
        raw_sims = torch.matmul(query_si_feat, tar_img_feat.t())
        sims = torch.sigmoid(raw_sims)
        evidences = torch.exp(torch.tanh(raw_sims)/self.tau)
        sims_tanh = torch.tanh(raw_sims)

        sum_e = evidences + evidences.t()
        norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
        alpha_i2t = evidences + 1
        alpha_t2i = evidences.t() + 1

        return alpha_i2t, alpha_t2i, norm_e, sims_tanh, sims



def KL(alpha, c):
    beta = torch.ones((1, c)).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(label, alpha, c, lkl_loss_weight, le_loss_weight):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    
    C = lkl_loss_weight * KL(alp, c)
    L_e = le_loss_weight * (A + B)
    return L_e + C


def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model


