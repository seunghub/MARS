import torch
import torch.nn as nn
import copy
from models.resnet18_se import resnet18_se
import math
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np

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

def modality_combination(x_rgb, x_ir, x_depth):
    modality_combi = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).to(x_rgb.device)
    x_rgb_repeated = x_rgb.repeat_interleave(len(modality_combi), dim=0)
    x_ir_repeated = x_ir.repeat_interleave(len(modality_combi), dim=0)
    x_depth_repeated = x_depth.repeat_interleave(len(modality_combi), dim=0)
    modality_combi_repeated = modality_combi.repeat(len(x_rgb), *[1] * (modality_combi.dim() - 1))
    x_rgb_out = modality_combi_repeated[:,0][:,None,None,None]*x_rgb_repeated
    x_ir_out = modality_combi_repeated[:,1][:,None,None,None]*x_ir_repeated
    x_depth_out = modality_combi_repeated[:,2][:,None,None,None]*x_depth_repeated

    mod_combi = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).to(x_rgb.device).float()
    mod_combi = mod_combi.repeat(x_rgb.shape[0],1)
    return x_rgb_out, x_ir_out, x_depth_out, mod_combi

def modality_dropout(x_rgb, x_ir, x_depth, prob):

    modality_combination = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    index_list = [x for x in range(6)]

    indices = np.random.choice(index_list, size=x_rgb.shape[0], replace=True, p=prob)
    p = modality_combination[indices]

    p = np.array(p)
    p = torch.from_numpy(p).float().to(x_rgb.device)

    Mask = p[...,None,None,None]

    x_rgb_sampled = x_rgb * Mask[:, 0]
    x_ir_sampled = x_ir * Mask[:, 1]
    x_depth_sampled = x_depth * Mask[:, 2]
    
    return x_rgb_sampled, x_ir_sampled, x_depth_sampled, p

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

def create_one_hot_batch(batch_size, num_classes=7):
    indices = torch.arange(batch_size) % num_classes
    one_hot = nn.functional.one_hot(indices, num_classes=num_classes).float()
    return one_hot

class OURMODEL(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)
        self.shared_bone = model_resnet18_se_1.layer3_new
        self.mod_comb_projector = nn.Linear(3, 16)
        self.gating = NoisyGate(256, args.num_experts, args.top_k)
        self.moe_layer = MOE(model_resnet18_se_1.layer4, args.num_experts)
        self.target_classifier = nn.Linear(512, args.class_num)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.args = args

        self.prob = np.array((1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6))

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb_full = self.special_bone_rgb(img_rgb)
        x_ir_full = self.special_bone_ir(img_ir)
        x_depth_full = self.special_bone_depth(img_depth)
        x_full = torch.cat((x_rgb_full, x_ir_full, x_depth_full), dim=1)


        # x_rgb, x_ir, x_depth = modality_combination(x_rgb, x_ir, x_depth)
        if self.training: x_rgb, x_ir, x_depth, mod_combi = modality_dropout(x_rgb_full, x_ir_full, x_depth_full, self.prob)
        else: x_rgb, x_ir, x_depth, mod_combi = modality_combination(x_rgb_full, x_ir_full, x_depth_full)
        
        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)

        features = self.shared_bone(x)
        anchor_feature = self.shared_bone(x_full)

        if self.training:
            residual=features-anchor_feature

            ## FULL MODALITY TASK LOSS
            topk_prob_full, _,_,_,_,_ = self.gating(anchor_feature, None)
            pred_label_full = self.target_classifier(self.pooling((self.moe_layer(anchor_feature) * topk_prob_full[...,None,None,None]).sum(1)).view(anchor_feature.shape[0], -1))
        else:
            residual=None
            pred_label_full=None
        
        topk_prob, load, importance, gating_distill_loss, var_order_loss, mismatch_vec = self.gating(features, residual)

        features = self.moe_layer(features)
        features = (features * topk_prob[...,None,None,None]).sum(1)
        features_ = self.pooling(features)
        features_ = features_.view(features_.shape[0], -1)
        pred_label = self.target_classifier(features_)

        return pred_label, pred_label_full, load, importance, gating_distill_loss, var_order_loss, mod_combi, mismatch_vec, topk_prob
