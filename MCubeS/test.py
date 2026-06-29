import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from PIL import Image


LABEL_COLORS_NEW_EN = {
    "#2ca02c" : "asphalt",      #0
    "#1f77b4" : "concrete",     #1
    "#ff7f0e" : "metal",        #2
    "#d62728" : "road marking", #3
    "#8c564b" : "fabric, leather",#4
    "#7f7f7f" : "glass",        #5
    "#bcbd22" : "plaster",      #6
    "#ff9896" : "plastic",      #7
    "#17becf" : "rubber",#8
    "#aec7e8" : "sand",         #9
    "#c49c94" : "gravel",       #10
    "#c5b0d5" : "ceramic",      #11
    "#f7b6d2" : "cobblestone",  #12
    "#c7c7c7" : "brick",        #13
    "#dbdb8d" : "grass",        #14
    "#9edae5" : "wood",         #15
    "#393b79" : "leaf",         #16
    "#6b6ecf" : "water",        #17
    "#9c9ede" : "human body",   #18
    "#637939" : "sky"}          #19


        

        
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

        
class TesterMultimodal(object):
    def __init__(self, args):
        self.args = args

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(f'{os.path.dirname(args.pth_path)}/test')
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        input_dim = 3
        
        model = DeepLabMultiInput(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        input_dim=input_dim,
                        ratio=args.ratio,
                        pretrained=args.use_pretrained_resnet,
                        num_experts=args.num_experts,
                        top_k=args.top_k)
        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        
        # Define Optimizer
        optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,lr=args.lr,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=0).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Load model parameters
        checkpoint = torch.load(args.pth_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            print(checkpoint['epoch'])
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0



    def test(self, epoch=0):
        self.model.eval()
        self.evaluator.reset()

        # ----- config -----
        n_comb = 8
        combi_nir_mask = torch.tensor([0,0,0,1,0,1,1,1])
        num_chunks = 1 if len(combi_nir_mask) <= 1 else min(3, len(combi_nir_mask))
        combi_nir_mask_splits = torch.tensor_split(combi_nir_mask, num_chunks)


        mod_names = [
            "rgb",
            "rgb+aolp",
            "rgb+dolp",
            "rgb+nir",
            "rgb+aolp+dolp",
            "rgb+aolp+nir",
            "rgb+dolp+nir",
            "rgb+aolp+dolp+nir"
        ]
        per_k_evals = [Evaluator(self.nclass) for _ in range(n_comb)]

        # ----- save root -----
        save_root = os.path.join(os.path.dirname(self.args.pth_path), "test_images")
        os.makedirs(save_root, exist_ok=True)

        color_hex = list(LABEL_COLORS_NEW_EN.keys())
        color_rgb = [tuple(int(h[1+i:3+i], 16) for i in (0, 2, 4)) for h in color_hex]
        palette = np.array(color_rgb, dtype=np.uint8)  # (20, 3)

        mod_names = [
            "rgb",
            "rgb+aolp",
            "rgb+dolp",
            "rgb+nir",
            "rgb+aolp+dolp",
            "rgb+aolp+nir",
            "rgb+dolp+nir",
            "rgb+aolp+dolp+nir"
        ]

        tbar = tqdm(self.test_loader, desc="\r")
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            # ----------------- load batch -----------------
            image, target, aolp, dolp, nir, nir_mask, u_map, v_map, mask = \
                sample['image'], sample['label'], sample['aolp'], sample['dolp'], \
                sample['nir'], sample['nir_mask'], sample['u_map'], sample['v_map'], sample['mask']

            batch_size = image.shape[0]

            if self.args.cuda:
                image, target, aolp, dolp, nir, nir_mask = \
                    image.cuda(), target.cuda(), aolp.cuda(), dolp.cuda(), nir.cuda(), nir_mask.cuda()

            if image.dim() == 3: image = image.unsqueeze(1)
            if aolp.dim() == 3: aolp = aolp.unsqueeze(1)
            if dolp.dim() == 3: dolp = dolp.unsqueeze(1)
            if nir.dim() == 3: nir = nir.unsqueeze(1)

            all_chunk_preds = []
            all_chunk_targets = []

            # ----------------- chunk inference -----------------
            for chunk in range(num_chunks):
                with torch.no_grad():
                    outputs = self.model(image, aolp, dolp, nir, num_chunks, chunk)
                    output = outputs[0]
                    Mask_nir_mask = combi_nir_mask_splits[chunk].repeat(len(image), *[1] * 0)[..., None, None].to(image.device)
                    tgt = target.repeat_interleave(len(combi_nir_mask_splits[chunk]), dim=0)
                    loss = self.criterion(output, tgt, nir_mask.repeat_interleave(len(combi_nir_mask_splits[chunk]), dim=0) * Mask_nir_mask)
                test_loss += loss.item()

                pred = output.argmax(dim=1).detach().cpu().numpy()
                tgt_np = tgt.detach().cpu().numpy()

                all_chunk_preds.append(pred)
                all_chunk_targets.append(tgt_np)

            # ----------------- concat chunks -----------------
            all_preds = np.concatenate(all_chunk_preds, axis=0)
            all_targets = np.concatenate(all_chunk_targets, axis=0)
            self.evaluator.add_batch(all_targets, all_preds)


            Btot = all_preds.shape[0]
            assert Btot % n_comb == 0, f"Batch {Btot} not divisible by n_comb={n_comb}"
            for k in range(n_comb):
                idx = np.arange(k, Btot, n_comb)  # k, k+8, k+16, ...
                per_k_evals[k].add_batch(all_targets[idx], all_preds[idx])

            # # ----------------- save per-sample -----------------
            # img_np = image.detach().cpu().numpy()  # (B, C, H, W)
            # if img_np.ndim == 3:
            #     img_np = np.expand_dims(img_np, 0)

            # B_total = all_preds.shape[0]
            # assert B_total % n_comb == 0, f"Batch {B_total} not divisible by n_comb={n_comb}"

            # per_sample_count = B_total // n_comb
            # for b in range(per_sample_count):
            #     sample_dir = os.path.join(save_root, f"sample_{i*batch_size + b:04d}")
            #     os.makedirs(sample_dir, exist_ok=True)

            #     # ---- RGB ----
            #     rgb = img_np[b].transpose(1, 2, 0)

            #     mean = np.array([0.485, 0.456, 0.406])
            #     std  = np.array([0.229, 0.224, 0.225])
            #     rgb = (rgb * std + mean) * 255.0

            #     rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            #     Image.fromarray(rgb).save(os.path.join(sample_dir, "input.png"))

            #     # ---- GT (Ground Truth) ----
            #     gt_img = all_targets[b].astype(np.int32)
            #     gt_img = np.clip(gt_img, 0, len(palette)-1)
            #     color_gt = palette[gt_img]
            #     Image.fromarray(color_gt).save(os.path.join(sample_dir, "gt.png"))

            #     for k, mod_name in enumerate(mod_names):
            #         idx = b + per_sample_count * k
            #         pred_img = all_preds[idx].astype(np.int32)
            #         pred_img = np.clip(pred_img, 0, len(palette)-1)
            #         color_pred = palette[pred_img]
            #         Image.fromarray(color_pred).save(os.path.join(sample_dir, f"{mod_name}.png"))

            # tbar.set_description(f"[{i+1}/{len(self.test_loader)}] Saved batch of {batch_size} samples")

        # ----------------- eval result -----------------
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        confusion_matrix = self.evaluator.confusion_matrix
        np.save(os.path.join(save_root, "confusion_matrix.npy"), confusion_matrix)


        per_k_mIoU = [ev.Mean_Intersection_over_Union() for ev in per_k_evals]
        print("\n=== Per-modality mIoU ===")
        for name, val in zip(mod_names, per_k_mIoU):
            print(f"{name:>18}: {val:.4f}")


        print("\n=== Test Results ===")
        print(f"Acc: {Acc:.4f}, Acc_class: {Acc_class:.4f}, mIoU: {mIoU:.4f}, FWIoU: {FWIoU:.4f}")
        print(f"Images saved in: {save_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'xception_adv','resnet_condconv'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'kitti', 'kitti_advanced', 'kitti_advanced_manta', 'handmade_dataset', 'handmade_dataset_stereo', 'multimodal_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--ratio', type=float, default=None, metavar='N',
                        help='number of ratio in RGFSConv (default: 1)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # propagation and positional encoding option
    parser.add_argument('--propagation', type=int, default=0,
                        help='image propagation length (default: 0)')
    parser.add_argument('--positional-encoding', action='store_true', default=False,
                        help='use positional encoding')
    parser.add_argument('--use-aolp', action='store_true', default=False,
                        help='use aolp')
    parser.add_argument('--use-dolp', action='store_true', default=False,
                        help='use dolp')
    parser.add_argument('--use-nir', action='store_true', default=False,
                        help='use nir')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--list-folder', type=str, default='list_folder1')
    parser.add_argument('--is-multimodal', action='store_true', default=False,
                        help='use multihead architecture')
    parser.add_argument('--pth-path', type=str, default=None,
                        help='set the pth file path')
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=5)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'kitti': 50,
            'kitti_advanced': 50
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'kitti' : 0.01,
            'kitti_advanced' : 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)

    seed_torch(args.seed)

    if args.is_multimodal:
        print("USE Multimodal Model")
        tester = TesterMultimodal(args)
    # else:
    #     tester = TesterAdv(args)

    print('Starting Epoch:', tester.args.start_epoch)
    print('Total Epoches:', tester.args.epochs)
    tester.test()
    tester.writer.close()
    print(args)
