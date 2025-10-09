import os
import torch
import logging
import time
import sys
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np

import torch.nn.functional as F

class Partitioner:
    
    def __init__(self, type, split, threshold=0.5, timestamp=None, epoch=None, dataset_name=None):
        self.type = type
        self.split = split
        self.threshold = threshold
        # self.debug = debug
        self.timestamp = timestamp
        self.epoch = epoch
        self.dataset_name = dataset_name
        
    def fit_features(self, model, trainloader, fabric, debug=False):
        dataset = trainloader.dataset
        if self.type == 'all_positive':
            logging.info('no partition, all positive')
            return torch.ones(len(dataset)) # all clean
        logging.info('fitting partitioner...')
        model.eval()
        data_size = len(dataset)
        loss = torch.zeros(data_size)
        sim = torch.zeros(data_size)
        
        for batch in tqdm(trainloader, desc="fitting partitioner"):  
            with torch.cuda.amp.autocast(): 
                l, s = model.per_loss(batch, fabric)
            for b in range(l.size(0)):
                loss[batch['pair_id'][b]] = l[b]
                sim[batch['pair_id'][b]] = s[b]
        
            
        self.losses = (loss-loss.min())/(loss.max()-loss.min())
        self.sims = (sim-sim.min())/(sim.max()-sim.min())
        self.pred = self.get_pred(self.type, debug=debug)
        return self.pred
        
     
    def get_pred(self, type, threshold=None, debug=False):
        type = type.lower()
        if threshold is None:
            threshold = self.threshold
        if type.lower() == 'gmm':
            logging.info('type.lower() == gmm')
            input_loss = self.losses.reshape(-1,1) 
            input_sim = self.sims.reshape(-1,1)
            input_data = input_loss if self.split == 'loss' else input_sim
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_data.cpu().numpy())
            clean_component_idx = gmm.means_.argmin() if self.split == 'loss' else gmm.means_.argmax()
            self.prob = torch.Tensor(gmm.predict_proba(input_data.cpu().numpy())[:, clean_component_idx])
            self.pred = (self.prob > threshold) + 0
            if debug:
                save_path = f'partitioner_log/{self.timestamp}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.losses, f'{save_path}/loss_{self.epoch}.pth')
                torch.save(self.sims, f'{save_path}/sim_{self.epoch}.pth')
                torch.save(self.prob, f'{save_path}/prob_{self.epoch}.pth')
                exit(0)
            area_num = torch.histc(torch.tensor(self.prob), bins=10, min=0.0, max=1.0).to(torch.int).tolist()
            logging.info(f'The counts in the equal areas are: {area_num}')
            clean_pro = self.pred.sum().item() / self.pred.shape[0]
            logging.info(f'the proportion of clean samples are {clean_pro}')
            return self.pred
        elif type == 'direct':
            if self.split == 'loss':
                input_data = self.losses
            elif self.split == 'sim':
                input_data = self.sims
            else:
                raise ValueError(f"the parameter split is invalid.")
            self.pred = (input_data < threshold) + 0
            self.prob = self.pred
            print('the proportion of clean samples are ', self.pred.sum().item() / self.pred.shape[0])
            return self.pred
        elif type == 'percent':
            if self.split == 'loss':
                input_data = self.losses
            elif self.split == 'sim':
                input_data = self.sims
            else:
                raise ValueError(f"the parameter split is invalid.")
            noisy_indices = input_data.argsort(descending=True)[:int(threshold * input_data.shape[0])]
            self.pred = torch.ones_like(input_data)
            self.pred[noisy_indices] = 0
            self.prob = self.pred
            print('the proportion of clean samples are ', self.pred.sum().item() / self.pred.shape[0])
            return self.pred
        else:
            raise ValueError(f"the parameter type is invalid.")
        
    def get_prob(self):
        if self.prob is None:
            raise KeyError('prob does not exist')
        else:
            return self.prob


@torch.no_grad()
def get_correspondence(model, data_loader, epoch):
    model.eval()
    data_size = len(data_loader.dataset)
    preds = torch.zeros(data_size)
    labels = torch.zeros(data_size)
    uncertainty = torch.zeros(data_size)
    uncertainty1 = torch.zeros(data_size)
    uncertainty2 = torch.zeros(data_size)
    norm_es_eye = torch.zeros(data_size)

    print(f"=> Get predicted correspondence labels at epoch: {epoch}")
    for batch_idx, batch in enumerate(data_loader):  
        with torch.no_grad():
            tar_img_feat = batch["tar_img_feat"]
            k = tar_img_feat.size(0)
            device = model.device
            # Encode the target image
            tar_img_feat = tar_img_feat.to(device)
            tar_img_feat = F.normalize(tar_img_feat, dim=-1)
            # Compute the query image
            query_si_feat = model.compute_query_embs(batch)
            # CEL
            alpha, alpha_, norm_e, _, _ = model.get_alpha(query_si_feat, tar_img_feat)

            g_t = torch.from_numpy(np.array([i for i in range(k)])).to(device)
            pred = g_t.eq(torch.argmax(norm_e, dim=1)) + 0
            u_1 = k / torch.sum(alpha, dim=1, keepdim=True)
            u_2 = k / torch.sum(alpha_, dim=1, keepdim=True)
            u = u_1 + u_2
            for b in range(k):
                preds[batch['pair_id'][b]] = pred[b]
                norm_es_eye[batch['pair_id'][b]] = norm_e[b][b]
                uncertainty[batch['pair_id'][b]] = u[b]
                uncertainty1[batch['pair_id'][b]] = u_1[b]
                uncertainty2[batch['pair_id'][b]] = u_2[b]
                # labels[batch['pair_id'][b]] = ls[b]
                
        if batch_idx % 50 == 0:
            print(f"[{100.0 * batch_idx / len(data_loader):.0f}%]\t pred: {pred}")
            pass
            
    return preds.cpu(), norm_es_eye.cpu(), uncertainty.cpu()

