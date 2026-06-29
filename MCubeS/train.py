# train.py

import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.deeplab import *
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


def cv_squared(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0])
    return x.float().var() / (x.float().mean() ** 2 + eps)


def seed_torch(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CuBLAS deterministic workspace
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TrainerMultimodal(object):
    def __init__(self, args):
        self.args = args

        # Saver & Summary
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Model
        input_dim = 3
        model = DeepLabMultiInput(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            input_dim=input_dim,
            ratio=args.ratio,
            pretrained=args.use_pretrained_resnet,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )

        train_params = [
            {"params": model.get_1x_lr_params(), "lr": args.lr},
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
        ]

        # Optimizer
        optimizer = torch.optim.SGD(
            train_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

        # Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), f"{args.dataset}_classes_weights.npy")
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Evaluator & Scheduler
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # CUDA
        if args.cuda:
            self.model = self.model.to(f"cuda:{args.gpu_ids[0]}")

        # Resume
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            if args.cuda:
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        if args.ft:
            args.start_epoch = 0

    # ---------------------------
    # Train
    # ---------------------------
    def training(self, epoch: int) -> None:
        train_loss = 0.0
        train_distill_loss = 0.0
        train_task_loss = 0.0

        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        combi_score_sum = 0
        combi_score_count = 0

        scaler = torch.amp.GradScaler("cuda")

        for i, sample in enumerate(tbar):
            image, target, aolp, dolp, nir, nir_mask, u_map, v_map, mask = (
                sample["image"], sample["label"], sample["aolp"], sample["dolp"],
                sample["nir"], sample["nir_mask"], sample["u_map"], sample["v_map"], sample["mask"]
            )

            if image.dim() == 3:
                image = image.unsqueeze(1)
            if aolp.dim() == 3:
                aolp = aolp.unsqueeze(1)
            if dolp.dim() == 3:
                dolp = dolp.unsqueeze(1)
            if nir.dim() == 3:
                nir = nir.unsqueeze(1)

            if self.args.cuda:
                image, target, aolp, dolp, nir, nir_mask = (
                    image.cuda(), target.cuda(), aolp.cuda(),
                    dolp.cuda(), nir.cuda(), nir_mask.cuda()
                )

            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                (
                    output,
                    output_full,
                    load,
                    importance,
                    gating_distill_loss,
                    var_order_loss,
                    mismatch_vec,
                ) = self.model(image, aolp, dolp, nir)

                full_task_loss = self.criterion(output, target, nir_mask)
                partial_task_loss = self.criterion(output_full, target, nir_mask)
                load_balancing_loss = cv_squared(load) + cv_squared(importance)
                task_loss = full_task_loss + partial_task_loss

                loss = (
                    self.args.weight_task * task_loss
                    + self.args.weight_lb * load_balancing_loss
                    + self.args.weight_distill * gating_distill_loss.mean()
                    + self.args.weight_order * var_order_loss
                )

            # 7 combinations when training (without pure AoLP/DoLP-only etc.)
            idx = np.arange(image.size(0)) % 7
            batch_score_sums = np.bincount(idx, weights=mismatch_vec.cpu().numpy(), minlength=7)
            batch_score_counts = np.bincount(idx, minlength=7)

            combi_score_sum += batch_score_sums
            combi_score_count += batch_score_counts

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            train_loss += loss.item()
            train_distill_loss += gating_distill_loss.mean().item()
            train_task_loss += task_loss.item()

            tbar.set_description(
                f"[Epoch {epoch}] Task Loss: {train_task_loss / (i + 1):.3f}, "
                f"Distill Loss: {train_distill_loss / (i + 1):.3f}"
            )
            self.writer.add_scalar("train/total_loss_iter", loss.item(), i + num_img_tr * epoch)

            # Visualization
            if i % max(1, (num_img_tr // 10)) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image[0], target, output, global_step)

        self.writer.add_scalar("train/total_loss_epoch", train_loss / (i + 1), epoch)
        print(f"[Epoch: {epoch}, numImages: {i * self.args.batch_size + image[0].data.shape[0]}]")
        print(f"Loss: {train_loss / (i + 1):.3f} Prev Best: {self.best_pred:.4f}")

        # Update sampling prob after a burn-in
        if epoch >= self.args.prob_epoch:
            temperature = 0.06
            new_prob = np.where(combi_score_count != 0, combi_score_sum / combi_score_count, 0.0)
            self.model.prob = torch.Tensor(
                np.exp(new_prob / temperature) / np.exp(new_prob / temperature).sum()
            )
            print(self.model.prob)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
            )

    # ---------------------------
    # Validation
    # ---------------------------
    def validation(self, epoch: int) -> None:
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0

        # split 8 combos into chunks for memory
        num_chunks = 2
        combi_nir_mask = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1])
        combi_nir_mask_splits = torch.tensor_split(combi_nir_mask, num_chunks)

        for i, sample in enumerate(tbar):
            image, target, aolp, dolp, nir, nir_mask, u_map, v_map, mask = (
                sample["image"], sample["label"], sample["aolp"], sample["dolp"],
                sample["nir"], sample["nir_mask"], sample["u_map"], sample["v_map"], sample["mask"]
            )

            if self.args.cuda:
                image, target, aolp, dolp, nir, nir_mask = (
                    image.cuda(), target.cuda(), aolp.cuda(),
                    dolp.cuda(), nir.cuda(), nir_mask.cuda()
                )

            if image.dim() == 3:
                image = image.unsqueeze(1)
            if aolp.dim() == 3:
                aolp = aolp.unsqueeze(1)
            if dolp.dim() == 3:
                dolp = dolp.unsqueeze(1)
            if nir.dim() == 3:
                nir = nir.unsqueeze(1)

            for chunk in range(num_chunks):
                with torch.no_grad():
                    outputs = self.model(image, aolp, dolp, nir, num_chunks, chunk)
                    output = outputs[0]
                    mask_nir = combi_nir_mask_splits[chunk].repeat(len(image))[..., None, None].to(image.device)
                    if chunk == 0:
                        target = target.repeat_interleave(len(combi_nir_mask_splits[chunk]), dim=0)
                    loss = self.criterion(
                        output,
                        target,
                        nir_mask.repeat_interleave(len(combi_nir_mask_splits[chunk]), dim=0) * mask_nir,
                    )
                test_loss += loss.item()
                tbar.set_description(f"[Epoch {epoch}] Val loss: {test_loss / (i + 1):.3f}")

                target_ = target.cpu().numpy()
                pred = output.argmax(dim=1).detach().cpu().numpy()
                self.evaluator.add_batch(target_, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        self.writer.add_scalar("val/total_loss_epoch", test_loss / (i + 1), epoch)
        self.writer.add_scalar("val/mIoU", mIoU, epoch)
        self.writer.add_scalar("val/Acc", Acc, epoch)
        self.writer.add_scalar("val/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("val/fwIoU", FWIoU, epoch)

        self.summary.visualize_validation_image(self.writer, self.args.dataset, image[0], target, output, epoch)

        print("Validation:")
        print(f"[Epoch: {epoch}, numImages: {i * self.args.batch_size + image[0].data.shape[0]}]")
        print(f"Acc:{Acc:.4f}, Acc_class:{Acc_class:.4f}, mIoU:{mIoU:.4f}, fwIoU:{FWIoU:.4f}")
        print(f"Loss: {test_loss:.3f}")

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
            )

    # ---------------------------
    # Test
    # ---------------------------
    def test(self, epoch: int) -> None:
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc="\r")
        test_loss = 0.0

        num_chunks = 3
        combi_nir_mask = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1])
        combi_nir_mask_splits = torch.tensor_split(combi_nir_mask, num_chunks)

        for i, sample in enumerate(tbar):
            image, target, aolp, dolp, nir, nir_mask, u_map, v_map, mask = (
                sample["image"], sample["label"], sample["aolp"], sample["dolp"],
                sample["nir"], sample["nir_mask"], sample["u_map"], sample["v_map"], sample["mask"]
            )

            if self.args.cuda:
                image, target, aolp, dolp, nir, nir_mask = (
                    image.cuda(), target.cuda(), aolp.cuda(),
                    dolp.cuda(), nir.cuda(), nir_mask.cuda()
                )

            if image.dim() == 3:
                image = image.unsqueeze(1)
            if aolp.dim() == 3:
                aolp = aolp.unsqueeze(1)
            if dolp.dim() == 3:
                dolp = dolp.unsqueeze(1)
            if nir.dim() == 3:
                nir = nir.unsqueeze(1)

            for chunk in range(num_chunks):
                with torch.no_grad():
                    outputs = self.model(image, aolp, dolp, nir, num_chunks, chunk)
                    output = outputs[0]
                    mask_nir = combi_nir_mask_splits[chunk].repeat(len(image))[..., None, None].to(image.device)
                    if chunk == 0:
                        target = target.repeat_interleave(len(combi_nir_mask_splits[chunk]), dim=0)
                    loss = self.criterion(
                        output,
                        target,
                        nir_mask.repeat_interleave(len(combi_nir_mask_splits[chunk]), dim=0) * mask_nir,
                    )
                test_loss += loss.item()
                tbar.set_description(f"[Epoch {epoch}] Test loss: {test_loss / (i + 1):.3f}")

                target_ = target.cpu().numpy()
                pred = output.argmax(dim=1).detach().cpu().numpy()
                self.evaluator.add_batch(target_, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        self.writer.add_scalar("test/total_loss_epoch", test_loss / (i + 1), epoch)
        self.writer.add_scalar("test/mIoU", mIoU, epoch)
        self.writer.add_scalar("test/Acc", Acc, epoch)
        self.writer.add_scalar("test/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("test/fwIoU", FWIoU, epoch)

        self.summary.visualize_test_image(self.writer, self.args.dataset, image[0], target, output, epoch)

        print("Test:")
        print(f"[Epoch: {epoch}, numImages: {i * self.args.batch_size + image[0].data.shape[0]}]")
        print(f"Acc:{Acc:.4f}, Acc_class:{Acc_class:.4f}, mIoU:{mIoU:.4f}, fwIoU:{FWIoU:.4f}")
        print(f"Loss: {test_loss:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    parser.add_argument("--backbone", type=str, default="resnet_adv",
                        choices=["resnet", "xception", "drn", "mobilenet", "resnet_adv", "xception_adv", "resnet_condconv"])
    parser.add_argument("--out-stride", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="multimodal_dataset",
                        choices=["pascal", "coco", "cityscapes", "kitti", "kitti_advanced", "kitti_advanced_manta",
                                 "handmade_dataset", "handmade_dataset_stereo", "multimodal_dataset"])
    parser.add_argument("--use-sbd", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--sync-bn", type=bool, default=True)
    parser.add_argument("--freeze-bn", type=bool, default=False)
    parser.add_argument("--loss-type", type=str, default="ce",
                        choices=["ce", "focal", "original", "bce"])

    # training hyper params
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--use-balanced-weights", action="store_true", default=False)
    parser.add_argument("--ratio", type=float, default=None)

    # optimizer params
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-scheduler", type=str, default="poly", choices=["poly", "step", "cos"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true", default=False)

    # cuda, seed and logging
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)

    # checkpointing
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkname", type=str, default=None)

    # finetune
    parser.add_argument("--ft", action="store_true", default=False)

    # eval
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--no-val", action="store_true", default=False)

    # options
    parser.add_argument("--propagation", type=int, default=0)
    parser.add_argument("--positional-encoding", action="store_true", default=False)
    parser.add_argument("--use-aolp", action="store_true", default=False)
    parser.add_argument("--use-dolp", action="store_true", default=False)
    parser.add_argument("--use-nir", action="store_true", default=False)
    parser.add_argument("--use-pretrained-resnet", action="store_true", default=False)
    parser.add_argument("--list-folder", type=str, default="list_folder")
    parser.add_argument("--is-multimodal", action="store_true", default=False)

    # method hyperparams
    parser.add_argument("--weight_task", type=float, default=1.0)
    parser.add_argument("--weight_lb", type=float, default=1.0)
    parser.add_argument("--weight_distill", type=float, default=1.0)
    parser.add_argument("--weight_order", type=float, default=1.0)
    parser.add_argument("--prob_epoch", type=int, default=1000)
    parser.add_argument("--name", type=str, default="test_result")
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    # CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError("Argument --gpu_ids must be a comma-separated list of integers only")

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # defaults based on dataset (if needed)
    if args.epochs is None:
        epoches = {"coco": 30, "cityscapes": 200, "pascal": 50, "kitti": 50, "kitti_advanced": 50}
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {"coco": 0.1, "cityscapes": 0.01, "pascal": 0.007, "kitti": 0.01, "kitti_advanced": 0.01}
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = f"deeplab-{args.backbone}"

    print(args)

    seed_torch(args.seed)

    # Trainer
    if args.is_multimodal:
        print("USE Multimodal Model")
        trainer = TrainerMultimodal(args)
    else:
        raise NotImplementedError("Only multimodal trainer is wired in this script.")

    print("Starting Epoch:", trainer.args.start_epoch)
    print("Total Epoches:", trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            if epoch > 420:  # original condition kept
                trainer.validation(epoch)
            # trainer.validation(epoch)

    trainer.writer.close()
    print(args)
