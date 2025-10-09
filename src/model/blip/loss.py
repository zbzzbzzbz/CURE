import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Hard Negative NCE loss for contrastive learning.
    """

    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        print("loss_func : CrossEntropyLoss")

    def forward(self, tar_img_feat: torch.Tensor, query_feat: torch.Tensor, temp, labels):
        device = tar_img_feat.device

        sim_t2q = tar_img_feat @ query_feat.T / temp
        sim_q2t = query_feat @ tar_img_feat.T / temp

        bs = sim_t2q.size(0)
        loss_t2q = F.cross_entropy(sim_t2q, torch.arange(bs, device=device))
        loss_q2t = F.cross_entropy(sim_q2t, torch.arange(bs, device=device))

        return (loss_t2q + loss_q2t) / 2


class HardNegativeNCE(nn.Module):
    """
    Hard-Negative NCE loss for contrastive learning.
    https://arxiv.org/pdf/2301.02280.pdf
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        print("loss_func : HardNegativeNCE")

    def forward(
        self,
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        temp,
        labels
    ):
        """
        Args:
            video_embds: (batch_size, video_embd_dim)
            text_embds: (batch_size, text_embd_dim)
        """
        batch_size = video_embds.size(0)
        # computation of the similarity matrix
        sim_matrix = video_embds @ text_embds.T  # (batch_size, batch_size)
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / temp
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss
    

class Robust_infoNCE(nn.Module):
    def __init__(self, **kwargs):
        super(Robust_infoNCE, self).__init__()
        print("loss_func : Robust_infoNCE")
        
    def forward(
        self,
        query_feat: torch.Tensor,
        tar_img_feat: torch.Tensor,
        temp,
        labels
    ):
        eps=1e-7        
        sim_q2t = query_feat @ tar_img_feat.T / temp
        
        i2t = (sim_q2t).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        target=torch.arange(sim_q2t.shape[0]).to(sim_q2t.device)
        clean_mask = labels.to(bool)
        noise_mask = ~clean_mask
        ploss = get_closs(i2t[clean_mask], target[clean_mask], 'RCL')
        nloss = get_closs(i2t[noise_mask], target[noise_mask], 'RCL')
        trade_off = 0.8
        return trade_off * ploss + (1 - trade_off) * nloss


def get_closs(i2t, target, loss_name=None):
    loss = torch.tensor(0.).to(i2t.device)
    bs = i2t.shape[0]
    if bs == 0:
        return loss
    if loss_name == 'None' or loss_name is None:
        return loss
    if loss_name == 'RCL':
        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), target] = 0.
        loss = - ((1. - i2t).log() * mask).sum() / bs
        return loss
    if loss_name == 'infoNCE':
        mask = torch.zeros_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), target] = 1.
        loss = - (i2t.log() * mask).sum() / bs
        return loss
    raise ValueError('loss name is invalid')


class TAL(nn.Module):
    def __init__(self, margin: float = 0.1, delta = 10, p_threshold: float = 0.5, **kwargs):
        super(TAL, self).__init__()
        self.margin = margin
        self.delta = delta
        self.p_threshold = p_threshold
        print("loss_func : TAL")
        print(f"self.margin : {self.margin} | self.p_threshold : {self.p_threshold}")
        
    def forward(
        self,
        query_feat: torch.Tensor,
        tar_img_feat: torch.Tensor,
        temp,
        labels,
        uncertainty=None,
        m = 10,
    ):
        scores = query_feat @ tar_img_feat.T
        batch_size = scores.shape[0]
        labels_pos = torch.eye(batch_size, device=scores.device, dtype=torch.float32)
        mask = 1 - labels_pos
        exp_scores = (scores / temp).exp()
        pos_scores = torch.diag(scores)
        
        neg_sum_i2t = (exp_scores * mask).sum(dim=1).clamp(max=10e35)
        loss_i2t = (- pos_scores + temp * neg_sum_i2t.log() + self.margin).clamp(min=0)

        neg_sum_t2i = (exp_scores.t() * mask).sum(dim=1).clamp(max=10e35)
        loss_t2i = (- pos_scores + temp * neg_sum_t2i.log() + self.margin).clamp(min=0)  
        
        total_loss = loss_i2t + loss_t2i

        if uncertainty is not None:

            uncertainty = uncertainty / 2
            is_uncertain = (uncertainty >= self.p_threshold)

            x = 2 * uncertainty - 1
            term_clean = 1 - (torch.pow(m, x) - 1) / (m - 1)
            soft_margin_clean = term_clean * self.margin
            
            loss_i2t_soft_clean = (- pos_scores + temp * neg_sum_i2t.log() + soft_margin_clean).clamp(min=0)
            loss_t2i_soft_clean = (- pos_scores + temp * neg_sum_t2i.log() + soft_margin_clean).clamp(min=0)
            total_loss_soft_clean = loss_i2t_soft_clean + loss_t2i_soft_clean

            labels_bool = torch.zeros_like(is_uncertain, dtype=torch.bool)
            labels_bool[labels] = True

            mask_clean_certain = labels_bool & (~is_uncertain)
            mask_clean_uncertain_clean = labels_bool & is_uncertain

            total_loss = torch.sum(total_loss[mask_clean_certain]) + torch.sum(total_loss_soft_clean[mask_clean_uncertain_clean])
            return total_loss
        else:
            total_loss = total_loss[labels]
            return torch.sum(total_loss)