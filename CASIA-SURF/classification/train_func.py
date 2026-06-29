import time
import os
import torch.nn as nn
import csv
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
from tqdm import tqdm
from test_func import calc_ACER_multi, batch_test
from torch.serialization import add_safe_globals
from argparse import Namespace
import global_var

add_safe_globals([Namespace])

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, args, optimizer, multiplier, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = args.total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0])
    return x.float().var() / (x.float().mean() ** 2 + eps)

def train_model_multi(model, criterion, optimizer, train_loader, test_loader, args):
    print(args)

    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    epoch = 0
    acer_best = 1
    log_list = []
    
    sampling_prob = []

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir, weights_only=True)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        global_var.epoch = epoch
        train_anchor_loss= 0
        train_label_loss = 0
        train_modality_loss = 0
        train_gate_distill_loss = 0
        train_var_order_loss = 0
        prev_prob = model.prob
        
        combi_score_sum = np.zeros(6)
        combi_score_count = np.zeros(6)    
        
        for batch_idx, batch_sample in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):
            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.to(f'cuda:{args.gpu}')
                img_ir = img_ir.to(f'cuda:{args.gpu}')
                img_depth = img_depth.to(f'cuda:{args.gpu}')
                target = target.to(f'cuda:{args.gpu}')

            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            pred_label, pred_label_full, load, importance, gate_distill_loss, var_order_loss, mod_combi, mismatch_vec, topk_prob = model(img_rgb, img_ir, img_depth)
            label_loss = criterion(pred_label,target)
            anchor_label_loss = criterion(pred_label_full,target)

            gate_loss = cv_squared(importance) + cv_squared(load)
            loss = args.anchor_task * anchor_label_loss.mean() + args.sampled_task * label_loss.mean() + args.load_balance* gate_loss.mean() + args.gate_distill * gate_distill_loss.mean() + args.var_order* var_order_loss

            train_gate_distill_loss += gate_distill_loss.mean().item()
            train_anchor_loss += anchor_label_loss.mean().item()
            train_label_loss += label_loss.mean().item()
            train_modality_loss += gate_loss.mean().item()
            train_var_order_loss += var_order_loss.item()

            mod_combi = mod_combi.int()
            codes = (mod_combi[:,0] << 2) + (mod_combi[:,1] << 1) + mod_combi[:,2]   # [4,2,1,6,5,3,7]

            # 우리가 원하는 패턴 순서: [100, 010, 001, 110, 101, 011]
            pattern_order = torch.tensor([4, 2, 1, 6, 5, 3], device=f'cuda:{args.gpu}')

            # numpy로 변환 후 합산
            # batch_score_sums = torch.bincount(codes, weights=gate_distill_loss, minlength=7)[pattern_order]
            batch_score_sums = torch.bincount(codes, weights=mismatch_vec, minlength=7)[pattern_order]
            # 개수
            batch_score_counts = torch.bincount(codes, minlength=7)[pattern_order]
            # numpy 변환 (CPU로)
            combi_score_sum += batch_score_sums.detach().cpu().numpy()
            combi_score_count += batch_score_counts.detach().cpu().numpy()


            loss.backward()
            optimizer.step()

        if epoch >= args.prob_epoch:
            # EMA
            alpha = args.alpha
            new_prob = np.where(combi_score_count != 0, combi_score_sum / combi_score_count, 0.0)
            # new_prob = new_prob / new_prob.sum()
            new_prob = np.exp(new_prob/args.temperature) / np.exp(new_prob/args.temperature).sum()
            prev_prob = model.prob
            update_prob = np.maximum(alpha * new_prob + (1-alpha) * prev_prob, 0.03)
            model.prob = update_prob / update_prob.sum()

            print([model.prob[i].item() for i in [0,2,1,4,3,5]])

        ##########################################################################################
        sampling_prob.append(model.prob)
        ##########################################################################################

        acer_test_list = calc_ACER_multi(model=model, loader=test_loader, args=args, verbose=True)
        acer_test = sum(acer_test_list) / len(acer_test_list)

        if acer_test < acer_best:
            acer_best = acer_test
            ##########################################################################################
            modality_combination = ['RGB','IR','Depth','RGB+IR','RGB+Depth','IR+Depth','RGB+IR+Depth']
            log_dir = args.log_root + '/' + args.name + '.csv'
            
            with open(log_dir, 'a+', newline='') as f:
                my_writer = csv.writer(f)
                for idx in [0,2,1,4,3,5,6]:
                    my_writer.writerow([f'{modality_combination[idx]:<12} : {acer_test_list[idx]:.4f}'])

            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)
            ##########################################################################################

        log_list.append(f'Epoch {epoch})')
        log_list.append(f' test_ACER: {acer_test.item():.4f}')
        log_list.append(f' best_ACER: {acer_best.item():.4f}')
        log_list.append(f' anchor_task: {train_anchor_loss / len(train_loader):.4f}')
        log_list.append(f' sampled_task: {train_label_loss / len(train_loader):.4f}')
        log_list.append(f' var_order: {train_var_order_loss / len(train_loader):.4f}')
        log_list.append(f' load_balance: {train_modality_loss / len(train_loader):.4f}')
        log_list.append(f' gate_distill: {train_gate_distill_loss / len(train_loader):.4f}')


        print(
            "Epoch {},ACER_test= {:.5f}, ACER_best= {:.5f}".format(epoch,acer_test, acer_best))
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)

        with open(log_dir, 'a+', newline='') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)

    sampling_prob = np.stack(sampling_prob, 0)
    np.save('sampling_prob.npy', sampling_prob)            
