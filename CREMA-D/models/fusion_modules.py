import torch
import torch.nn as nn
import copy
from torch.distributions.normal import Normal
import numpy as np
import math
import torch.nn.functional as F

def modality_combination(audio, video):
    modality_combi = torch.tensor([[1, 0], [0, 1], [1, 1]]).to(audio.device)
    audio_repeated = audio.repeat_interleave(len(modality_combi), dim=0)
    video_repeated = video.repeat_interleave(len(modality_combi), dim=0)
    modality_combi_repeated = modality_combi.repeat(len(audio), *[1] * (modality_combi.dim() - 1))

    audio_out = modality_combi_repeated[:,0][...,None]*audio_repeated
    video_out = modality_combi_repeated[:,1][...,None]*video_repeated

    mod_combi = torch.tensor([[1, 0], [0, 1], [1, 1]]).to(audio.device).float()
    mod_combi = mod_combi.repeat(audio.shape[0],1)
    return audio_out, video_out, mod_combi

def modality_dropout(audio, video, prob):

    modality_combination = np.array([[1, 0], [0, 1]])
    index_list = [x for x in range(2)]

    indices = np.random.choice(index_list, size=audio.shape[0], replace=True, p=prob)
    p = modality_combination[indices]

    p = np.array(p)
    p = torch.from_numpy(p).float().to(audio.device)

    audio_sampled = audio * p[:, 0].unsqueeze(-1)
    video_sampled = video * p[:, 1].unsqueeze(-1)
    
    return audio_sampled, video_sampled, p


def _variance_order_loss_sorted_adjacent_selected(s2_sorted, valid_counts):
    B, T = s2_sorted.shape
    diff = torch.log(s2_sorted[:, 1:]) - torch.log(s2_sorted[:, :-1])           # [B, T-1]
    # 각 배치에서 유효한 쌍 수: max(valid_counts-1, 0)
    idx = torch.arange(T - 1, device=s2_sorted.device).unsqueeze(0)  # [1, T-1]
    valid_pair_mask = idx < (valid_counts.clamp_min(0) - 1).unsqueeze(1)  # [B, T-1]
    diff_valid = diff[valid_pair_mask]

    if diff_valid.numel() == 0:                       
        return s2_sorted.new_tensor(0.0)
    
    return torch.mean(F.softplus(diff_valid))  

def _variance_order_loss_topk_union(clean_logits_res, clean_logits_fea, noise_stddev_res,
                                         top_k, tau=1.0, eps=1e-8):

    B, E = clean_logits_res.shape
    k_eff = min(2 * top_k, E) 

    p_t = torch.nn.functional.softmax(clean_logits_res / tau, dim=-1)
    p_s = torch.nn.functional.softmax(clean_logits_fea / tau, dim=-1)
    m_i = (p_t-p_s).detach().abs()  # [B,E]

    sigma2 = (noise_stddev_res ** 2).clamp_min(eps)           # [B,E]

    # union mask 
    _, top_idx_t = torch.topk(clean_logits_res, k=min(top_k, E), dim=-1)  # [B,K]
    _, top_idx_s = torch.topk(clean_logits_fea, k=min(top_k, E), dim=-1)  # [B,K]
    union_mask = torch.zeros((B, E), dtype=torch.bool, device=clean_logits_res.device)
    union_mask.scatter_(1, top_idx_t, True)
    union_mask.scatter_(1, top_idx_s, True)
    valid_counts = union_mask.sum(dim=-1)                       # [B]

    m_eff = m_i.masked_fill(~union_mask, float('-inf'))         # [B,E]
    top_vals, top_idx = torch.topk(m_eff, k=min(k_eff, E), dim=-1)  # [B,T]

    s2_sel = torch.gather(sigma2, -1, top_idx)                  # [B,T]

    loss = _variance_order_loss_sorted_adjacent_selected(s2_sel, valid_counts.clamp(max=k_eff))
    return loss

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output

class MOE(nn.Module):
    def __init__(self, expert, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            copy.deepcopy(expert) for _ in range(num_experts)
        ])

    def forward(self, x):
        outputs = []
        for idx in range(self.num_experts):
            expert_out = self.experts[idx](x)
            outputs.append(expert_out)
        outputs = torch.stack(outputs, dim=1)
        return outputs

class NoisyGate(nn.Module):
    def __init__(self, d_model, tot_expert, top_k):
        super().__init__()
        self.tot_expert = tot_expert
        self.pooling_res = nn.AdaptiveAvgPool2d((1,1))
        self.w_gate_res = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )
        self.w_noise_res = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )
        self.pooling_fea = nn.AdaptiveAvgPool2d((1,1))
        self.w_noise_fea = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )
        self.w_gate_fea = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.tot_expert)
        )
        self.top_k = top_k
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.noise_epsilon = 1e-2
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_gate_res, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise_res, a=math.sqrt(5))

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, feature, residual=None):
        if residual is not None:
            if residual.dim() > 3:
                residual = self.pooling_res(residual)
                residual = residual.view(residual.shape[0],-1)
            clean_logits_res = residual @ self.w_gate_res
            raw_noise_stddev_res = residual @ self.w_noise_res
            noise_stddev_res = (
                self.softplus(raw_noise_stddev_res) + self.noise_epsilon
            )
            if self.training: noisy_logits_res = clean_logits_res + (torch.randn_like(clean_logits_res) * noise_stddev_res)
            else: noisy_logits_res = clean_logits_res
        if feature.dim() > 3:
            feature = self.pooling_fea(feature)
            feature = feature.view(feature.shape[0],-1)
        clean_logits_fea = self.w_gate_fea(feature)

        if residual is not None:
            logits = noisy_logits_res
            clean_logits = clean_logits_res
            noisy_logits = noisy_logits_res
            noise_stddev = noise_stddev_res
        else:
            logits = clean_logits_fea

        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if residual is not None:
            if self.top_k < self.tot_expert:
                load = (
                    self._prob_in_top_k(
                        clean_logits, noisy_logits, noise_stddev, top_logits
                    )
                ).sum(0)
            else:
                load = (gates > 0).sum(0)
            importance = gates.sum(0)
        else:
            load = 0; importance=0

        if residual is not None:
            top_logits_res, top_indices_res = clean_logits_res.topk(min(self.top_k + 1, self.tot_expert), dim=1)
            top_k_logits_res = top_logits_res[:, : self.top_k]
            top_k_indices_res = top_indices_res[:, : self.top_k]
            top_k_gates_res = self.softmax(top_k_logits_res)
            zeros_res = torch.zeros_like(clean_logits_res, requires_grad=False)
            gates_res = zeros_res.scatter(1, top_k_indices_res, top_k_gates_res)

            _, top_t = clean_logits_res.topk(self.top_k, dim=1)   # [B,K], student clean
            _, top_s = clean_logits_fea.topk(self.top_k, dim=1)   # [B,K], student clean
            # inter = (top_indices_res.unsqueeze(-1) == top_s.unsqueeze(-2)).any(-1).float().sum(-1)  # [B]
            inter = (top_t.unsqueeze(-1) == top_s.unsqueeze(-2)).any(-1).float().sum(-1)  # [B]
            mismatch_vec = 1.0 - inter / float(self.top_k)

        if residual is None: gate_distill_loss = None; var_order_loss = None; mismatch_vec=None
        else: 
            gate_distill_loss = nn.functional.kl_div(torch.log_softmax(clean_logits_fea,dim=-1), nn.functional.softmax(clean_logits_res.clone().detach(),dim=-1), reduction='none').sum(-1)
            var_order_loss = _variance_order_loss_topk_union(clean_logits_res, clean_logits_fea, noise_stddev_res, self.top_k, tau=1.0, eps=1e-8)
        return (
            gates,
            load,
            importance,
            gate_distill_loss,
            var_order_loss,
            mismatch_vec
        )
    
class ConcatFusion_ours(nn.Module):
    def __init__(self, args, input_dim=1024, output_dim=100):
        super(ConcatFusion_ours, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )
        self.gating = NoisyGate(512, args.num_experts, args.top_k)
        expert_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )
        self.moe_layer = MOE(expert_layer, args.num_experts)
        self.args = args

    def forward(self, anchor_x, anchor_y, p):
        
        if self.training:
            x, y, mod_combi = modality_dropout(anchor_x,anchor_y,p)
        else:
            x, y, mod_combi = modality_combination(anchor_x,anchor_y)

        features = self.fusion_layer(torch.cat((x, y), dim=1))
        anchor_features = self.fusion_layer(torch.cat((anchor_x, anchor_y), dim=1))
        # features = self.fusion_layer(x+y)
        # anchor_features = self.fusion_layer(anchor_x+anchor_y)
        
        if self.training: 
            residual = features-anchor_features
            topk_prob_full, _,_,_,_,_ = self.gating(anchor_features, None)
            pred_label_full = self.fc_out((self.moe_layer(anchor_features) * topk_prob_full[...,None]).sum(1))
        else: 
            # residual = features-anchor_features.repeat_interleave(3, dim=0)
            residual = None
            pred_label_full=None

        topk_prob, load, importance, gating_distill_loss, var_order_loss, mismatch_vec = self.gating(features, residual)

        features = self.moe_layer(features)
        features = (features * topk_prob[...,None]).sum(1)
        pred_label = self.fc_out(features)

        return pred_label, pred_label_full, load, importance, gating_distill_loss, var_order_loss, mod_combi, mismatch_vec, topk_prob
    

class ConcatFusion(nn.Module):
    def __init__(self, args, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.auxi_fc = nn.Linear(input_dim, output_dim)

        self.mu_dul_backbone = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
        )
        self.args = args

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        if self.args.pme:

            mu_dul = self.mu_dul_backbone(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            epsilon = torch.randn_like(std_dul)

            if self.training:
                output = mu_dul + epsilon * std_dul
            else:
                output = mu_dul

            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        else:
            mu_dul = torch.zeros_like(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        return x, y, out, auxi_out, mu_dul, std_dul


class ConcatFusion_Swin(nn.Module):
    def __init__(self, args,input_dim=1024* 2, output_dim=100):
        super(ConcatFusion_Swin, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

        self.mu_dul_backbone = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Linear(input_dim,input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.args = args

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        if self.args.pme:

            mu_dul = self.mu_dul_backbone(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            epsilon = torch.randn_like(std_dul)

            if self.training:
                output = mu_dul + epsilon * std_dul
            else:
                output = mu_dul

            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        else:
            mu_dul = torch.zeros_like(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        return x, y, out, auxi_out, mu_dul, std_dul



class ConcatFusion_Vanilla(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion_Vanilla, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.auxi_fc = nn.Linear(input_dim, output_dim)
        #
        # self.mu_dul_backbone = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        # )
        # self.logvar_dul_backbone = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        # )
        # self.args=args

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        out = self.fc_out(output)
        auxi_out = self.auxi_fc(output)

        return x, y, out, auxi_out


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.fc = nn.Linear(512 * 512, 512)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):
        # if self.x_film:
        #     film = x
        #     to_be_film = y
        # else:
        #     film = y
        #     to_be_film = x
        #
        # gamma, beta = torch.split(self.fc(film), self.dim, 1)
        #
        # output = gamma * to_be_film + beta
        x = torch.unsqueeze(x, dim=2)
        y = torch.unsqueeze(y, dim=1)
        z = torch.bmm(x, y)
        # print(z.shape)
        z = z.view(z.shape[0], -1)
        output = self.fc(z)
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


