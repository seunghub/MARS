import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, ratio, input_heads=1):
        super(Decoder, self).__init__()

        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            last_conv_input = 304
        elif backbone == 'resnet_adv' or backbone == 'resnet_condconv':
            low_level_inplanes = 256 * input_heads
            last_conv_input = 256 * input_heads + 48
        elif backbone == 'xception':
            low_level_inplanes = 128
            last_conv_input = 304
        elif backbone == 'xception_adv':
            low_level_inplanes = 128 * input_heads
            last_conv_input = 256 * input_heads + 48
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
            last_conv_input = 304
        elif backbone == 'plus':
            low_level_inplanes = 256
            last_conv_input = 304
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def _variance_order_loss_sorted_adjacent_selected(s2_sorted, valid_counts):

    B, T = s2_sorted.shape
    diff = torch.log(s2_sorted[:, 1:]) - torch.log(s2_sorted[:, :-1])  # [B, T-1]

    idx = torch.arange(T - 1, device=s2_sorted.device).unsqueeze(0)
    valid_pair_mask = idx < (valid_counts.clamp_min(0) - 1).unsqueeze(1)  # [B, T-1]
    diff_valid = diff[valid_pair_mask]

    if diff_valid.numel() == 0:
        return s2_sorted.new_tensor(0.0)
    return torch.mean(F.softplus(diff_valid))


def _variance_order_loss_topk_union(
    clean_logits_res, clean_logits_fea, noise_stddev_res, top_k, tau=1.0, eps=1e-8
):

    B, E = clean_logits_res.shape
    k_eff = min(2 * top_k, E)

    p_t = torch.nn.functional.softmax(clean_logits_res / tau, dim=-1)
    p_s = torch.nn.functional.softmax(clean_logits_fea / tau, dim=-1)
    m_i = (p_t - p_s).detach().abs()  # [B, E]

    sigma2 = (noise_stddev_res ** 2).clamp_min(eps)  # [B, E]

    _, top_idx_t = torch.topk(clean_logits_res, k=min(top_k, E), dim=-1)
    _, top_idx_s = torch.topk(clean_logits_fea, k=min(top_k, E), dim=-1)
    union_mask = torch.zeros((B, E), dtype=torch.bool, device=clean_logits_res.device)
    union_mask.scatter_(1, top_idx_t, True)
    union_mask.scatter_(1, top_idx_s, True)
    valid_counts = union_mask.sum(dim=-1)

    m_eff = m_i.masked_fill(~union_mask, float('-inf'))
    _, top_idx = torch.topk(m_eff, k=min(k_eff, E), dim=-1)

    s2_sel = torch.gather(sigma2, -1, top_idx)
    loss = _variance_order_loss_sorted_adjacent_selected(s2_sel, valid_counts.clamp(max=k_eff))
    return loss


class MOE(nn.Module):

    def __init__(self, BatchNorm, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        outputs = []
        for idx in range(self.num_experts):
            expert_out = self.experts[idx](x)
            outputs.append(expert_out)
        outputs = torch.stack(outputs, dim=1)  # [B, E, C, H, W]
        return outputs


class NoisyGate(nn.Module):

    def __init__(self, last_conv_input, tot_expert, top_k):
        super().__init__()
        self.tot_expert = tot_expert
        self.top_k = top_k
        self.noise_epsilon = 1e-2

        # Residual branch
        self.pooling_res = nn.AdaptiveAvgPool2d((1, 1))
        self.w_gate_res = nn.Parameter(torch.zeros(last_conv_input, tot_expert), requires_grad=True)
        self.w_noise_res = nn.Parameter(torch.zeros(last_conv_input, tot_expert), requires_grad=True)

        # Feature branch
        self.pooling_fea = nn.AdaptiveAvgPool2d((1, 1))
        self.w_noise_fea = nn.Parameter(torch.zeros(last_conv_input, tot_expert), requires_grad=True)
        self.w_gate_fea = nn.Sequential(
            nn.Linear(last_conv_input, last_conv_input),
            nn.BatchNorm1d(last_conv_input),
            nn.ReLU(),
            nn.Linear(last_conv_input, tot_expert),
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_gate_res, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise_res, a=math.sqrt(5))

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)

        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(torch.tensor([0.0], device=clean_values.device), torch.tensor([1.0], device=clean_values.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, feature, residual=None):
        if residual is not None:
            if residual.dim() > 3:
                residual = self.pooling_res(residual)
                residual = residual.view(residual.shape[0], -1)

            clean_logits_res = residual @ self.w_gate_res
            raw_noise_stddev_res = residual @ self.w_noise_res
            noise_stddev_res = self.softplus(raw_noise_stddev_res) + self.noise_epsilon
            if self.training:
                noisy_logits_res = clean_logits_res + (torch.randn_like(clean_logits_res) * noise_stddev_res)
            else:
                noisy_logits_res = clean_logits_res

        if feature.dim() > 3:
            feature = self.pooling_fea(feature)
            feature = feature.view(feature.shape[0], -1)
        clean_logits_fea = self.w_gate_fea(feature)

        if residual is not None:
            logits = noisy_logits_res
            clean_logits = clean_logits_res
            noisy_logits = noisy_logits_res
            noise_stddev = noise_stddev_res
        else:
            logits = clean_logits_fea

        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.tot_expert), dim=1)
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits).to(top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if residual is not None:
            if self.top_k < self.tot_expert:
                load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
            else:
                load = (gates > 0).sum(0)
            importance = gates.sum(0)
        else:
            load = 0
            importance = 0

        if residual is not None:
            _, top_t = clean_logits_res.topk(self.top_k, dim=1)
            _, top_s = clean_logits_fea.topk(self.top_k, dim=1)
            inter = (top_t.unsqueeze(-1) == top_s.unsqueeze(-2)).any(-1).float().sum(-1)
            mismatch_vec = 1.0 - inter / float(self.top_k)

        if residual is None:
            gate_distill_loss = None
            var_order_loss = None
            mismatch_vec = None
        else:
            gate_distill_loss = nn.functional.kl_div(
                torch.log_softmax(clean_logits_fea, dim=-1),
                nn.functional.softmax(clean_logits_res.clone().detach(), dim=-1),
                reduction='none'
            ).mean(-1)
            var_order_loss = _variance_order_loss_topk_union(
                clean_logits_res, clean_logits_fea, noise_stddev_res, self.top_k, tau=1.0, eps=1e-8
            )

        return gates, load, importance, gate_distill_loss, var_order_loss, mismatch_vec


class OurDecoder(nn.Module):

    def __init__(self, num_classes, backbone, BatchNorm, ratio, input_heads=1, num_experts=16, top_k=5):
        super(OurDecoder, self).__init__()

        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            last_conv_input = 304
        elif backbone == 'resnet_adv' or backbone == 'resnet_condconv':
            low_level_inplanes = 256 * input_heads
            last_conv_input = 256 * input_heads + 48
        elif backbone == 'xception':
            low_level_inplanes = 128
            last_conv_input = 304
        elif backbone == 'xception_adv':
            low_level_inplanes = 128 * input_heads
            last_conv_input = 256 * input_heads + 48
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
            last_conv_input = 304
        elif backbone == 'plus':
            low_level_inplanes = 256
            last_conv_input = 304
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.shared_bone = nn.Sequential(
            nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.moe_layer = MOE(BatchNorm, num_experts)
        self.gating = NoisyGate(256, num_experts, top_k)
        self.head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, low_level_feat, x_full=None, low_level_feat_full=None):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        features = self.shared_bone(x)

        if x_full is not None:
            low_level_feat_full = self.conv1(low_level_feat_full)
            low_level_feat_full = self.bn1(low_level_feat_full)
            low_level_feat_full = self.relu(low_level_feat_full)

            x_full = F.interpolate(x_full, size=low_level_feat_full.size()[2:], mode='bilinear', align_corners=True)
            x_full = torch.cat((x_full, low_level_feat_full), dim=1)

            anchor_features = self.shared_bone(x_full)
            residual = features - anchor_features
        else:
            residual = None

        topk_prob, load, importance, gating_distill_loss, var_order_loss, mismatch_vec = self.gating(features, residual)
        features = self.moe_layer(features)
        features = (features * topk_prob[..., None, None, None]).sum(1)
        x = self.head(features)

        return x, load, importance, gating_distill_loss, var_order_loss, mismatch_vec

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm, ratio, input_heads=1, num_experts=16, top_k=5):
    # return Decoder(num_classes, backbone, BatchNorm, ratio, input_heads)
    return OurDecoder(num_classes, backbone, BatchNorm, ratio, input_heads, num_experts, top_k)
