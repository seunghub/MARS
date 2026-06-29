import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def modality_combination(
    x1, x2, x3, x4,
    low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4,
    num_chunks, chunk
):

    modality_combi_all = torch.tensor([
        [1, 0, 0, 0],
        # [0, 1, 0, 0],
        # [0, 0, 1, 0],
        # [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        # [0, 1, 1, 0],
        # [0, 1, 0, 1],
        # [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        # [0, 1, 1, 1],
        [1, 1, 1, 1]
    ], device=x1.device, dtype=torch.float)

    modality_combi = torch.tensor_split(modality_combi_all, num_chunks, dim=0)[chunk]

    # Repeat features
    x1_repeated = x1.repeat_interleave(len(modality_combi), dim=0)
    x2_repeated = x2.repeat_interleave(len(modality_combi), dim=0)
    x3_repeated = x3.repeat_interleave(len(modality_combi), dim=0)
    x4_repeated = x4.repeat_interleave(len(modality_combi), dim=0)

    low_level_feat1_repeated = low_level_feat1.repeat_interleave(len(modality_combi), dim=0)
    low_level_feat2_repeated = low_level_feat2.repeat_interleave(len(modality_combi), dim=0)
    low_level_feat3_repeated = low_level_feat3.repeat_interleave(len(modality_combi), dim=0)
    low_level_feat4_repeated = low_level_feat4.repeat_interleave(len(modality_combi), dim=0)

    modality_combi_repeated = modality_combi.repeat(len(x1), *[1] * (modality_combi.dim() - 1))

    x1_out = modality_combi_repeated[:, 0][:, None, None, None] * x1_repeated
    x2_out = modality_combi_repeated[:, 1][:, None, None, None] * x2_repeated
    x3_out = modality_combi_repeated[:, 2][:, None, None, None] * x3_repeated
    x4_out = modality_combi_repeated[:, 3][:, None, None, None] * x4_repeated

    low_level_feat1_out = modality_combi_repeated[:, 0][:, None, None, None] * low_level_feat1_repeated
    low_level_feat2_out = modality_combi_repeated[:, 1][:, None, None, None] * low_level_feat2_repeated
    low_level_feat3_out = modality_combi_repeated[:, 2][:, None, None, None] * low_level_feat3_repeated
    low_level_feat4_out = modality_combi_repeated[:, 3][:, None, None, None] * low_level_feat4_repeated

    mod_combi = modality_combi.repeat(x1.shape[0], 1)
    return (
        x1_out, x2_out, x3_out, x4_out,
        low_level_feat1_out, low_level_feat2_out, low_level_feat3_out, low_level_feat4_out,
        mod_combi
    )


def modality_dropout(
    x1, x2, x3, x4,
    low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4,
    prob
):

    modality_combi = torch.tensor([
        [1, 0, 0, 0],
        # [0, 1, 0, 0],
        # [0, 0, 1, 0],
        # [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        # [0, 1, 1, 0],
        # [0, 1, 0, 1],
        # [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        # [0, 1, 1, 1],
        # [1, 1, 1, 1]
    ], device=x1.device, dtype=torch.float)

    indices = torch.multinomial(prob, num_samples=x1.shape[0], replacement=True)
    mask = modality_combi[indices]

    x1_out = mask[:, 0][..., None, None, None] * x1
    x2_out = mask[:, 1][..., None, None, None] * x2
    x3_out = mask[:, 2][..., None, None, None] * x3
    x4_out = mask[:, 3][..., None, None, None] * x4

    low_level_feat1_out = mask[:, 0][..., None, None, None] * low_level_feat1
    low_level_feat2_out = mask[:, 1][..., None, None, None] * low_level_feat2
    low_level_feat3_out = mask[:, 2][..., None, None, None] * low_level_feat3
    low_level_feat4_out = mask[:, 3][..., None, None, None] * low_level_feat4

    return (
        x1_out, x2_out, x3_out, x4_out,
        low_level_feat1_out, low_level_feat2_out, low_level_feat3_out, low_level_feat4_out,
        mask
    )


class DeepLabMultiInput(nn.Module):
    def __init__(
        self, backbone='resnet', output_stride=16, num_classes=1,
        sync_bn=True, freeze_bn=False, input_dim=3, ratio=1,
        pretrained=False, num_experts=16, top_k=5
    ):
        super(DeepLabMultiInput, self).__init__()

        if backbone == 'drn':
            output_stride = 8

        BatchNorm = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        # Encoders
        self.backbone1 = build_backbone(backbone, output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)  # RGB
        self.aspp1 = build_aspp(backbone, output_stride, BatchNorm)

        self.backbone2 = build_backbone(backbone, output_stride, BatchNorm, input_dim=2)  # AoLP
        self.aspp2 = build_aspp(backbone, output_stride, BatchNorm)

        self.backbone3 = build_backbone(backbone, output_stride, BatchNorm, input_dim=1)  # DoLP
        self.aspp3 = build_aspp(backbone, output_stride, BatchNorm)

        self.backbone4 = build_backbone(backbone, output_stride, BatchNorm, input_dim=1)  # NIR
        self.aspp4 = build_aspp(backbone, output_stride, BatchNorm)

        # Decoder (ours)
        self.decoder = build_decoder(
            num_classes, backbone, BatchNorm, ratio, input_heads=4,
            num_experts=num_experts, top_k=top_k
        )

        self.freeze_bn = freeze_bn
        # self.prob = torch.full((14,), 1/14)
        self.prob = torch.full((7,), 1 / 7)

    def forward(self, input1, input2, input3, input4, num_chunks=None, chunk=None):
        # Encode
        x1_full, low_level_feat1_full = self.backbone1(input1)
        x1_full = self.aspp1(x1_full)

        x2_full, low_level_feat2_full = self.backbone2(input2)
        x2_full = self.aspp2(x2_full)

        x3_full, low_level_feat3_full = self.backbone3(input3)
        x3_full = self.aspp3(x3_full)

        x4_full, low_level_feat4_full = self.backbone4(input4)
        x4_full = self.aspp4(x4_full)

        if self.training:
            x1, x2, x3, x4, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4, mod_combi = \
                modality_dropout(
                    x1_full, x2_full, x3_full, x4_full,
                    low_level_feat1_full, low_level_feat2_full, low_level_feat3_full, low_level_feat4_full,
                    self.prob
                )
        else:
            x1, x2, x3, x4, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4, mod_combi = \
                modality_combination(
                    x1_full, x2_full, x3_full, x4_full,
                    low_level_feat1_full, low_level_feat2_full, low_level_feat3_full, low_level_feat4_full,
                    num_chunks, chunk
                )

        x = torch.cat([x1, x2, x3, x4], dim=1)
        low_level_feat = torch.cat([low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4], dim=1)

        if self.training:
        # if True: # ORACLE
            x_full = torch.cat([x1_full, x2_full, x3_full, x4_full], dim=1)
            low_level_feat_full = torch.cat(
                [low_level_feat1_full, low_level_feat2_full, low_level_feat3_full, low_level_feat4_full], dim=1
            )
        else:
            x_full = None
            low_level_feat_full = None

        x, load, importance, gating_distill_loss, var_order_loss, mismatch_vec = \
            self.decoder(x, low_level_feat, x_full, low_level_feat_full)

        out = F.interpolate(x, size=input1.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            x_full_out = self.decoder(x_full, low_level_feat_full)
            x_full_out = x_full_out[0]
            x_full_out = F.interpolate(x_full_out, size=input1.size()[2:], mode='bilinear', align_corners=True)
        else:
            x_full_out = None

        return out, x_full_out, load, importance, gating_distill_loss, var_order_loss, mismatch_vec

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone1, self.backbone2, self.backbone3, self.backbone4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp1, self.aspp2, self.aspp3, self.aspp4, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or \
                       isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                    if m[0] == 'gamma':
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = DeepLabMultiInput(backbone='mobilenet', output_stride=16)
    model.eval()
    _input = torch.rand(1, 3, 513, 513)
    out = model(_input, _input, _input, _input)
    print(out[0].shape)
