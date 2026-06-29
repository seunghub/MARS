import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from dataset.KSDataset import KSDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
import csv
import numpy as np



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=1, type=int)
    parser.add_argument('--audio_path', default='./data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')

    parser.add_argument('--ckpt_path', default='results/cramed/pme', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pme', type=int, default=1)

    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=5)
    
    parser.add_argument('--weight_task', type=float, default=1)
    parser.add_argument('--weight_lb', type=float, default=1)
    parser.add_argument('--weight_distill', type=float, default=1)
    parser.add_argument('--weight_order', type=float, default=1)

    parser.add_argument('--name', type=str, default='ours_default')
    parser.add_argument('--prob_epoch', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.)

    return parser.parse_args()


# def get_feature_diversity(a_feature):
#     a_feature = a_feature.view(a_feature.shape[0], a_feature.shape[1], -1)  # B C HW
#     a_feature = a_feature.permute(0, 2, 1)  # B HW C
#     a_feature = a_feature - torch.mean(a_feature, dim=2, keepdim=True)
#     a_similarity = torch.bmm(a_feature, a_feature.permute(0, 2, 1))
#     a_std = torch.std(a_feature, dim=2)
#     a_std_matrix = torch.bmm(a_std.unsqueeze(dim=2), a_std.unsqueeze(dim=1))
#     a_similarity = a_similarity / a_std_matrix
#     # print(a_similarity)
#     a_norm = torch.norm(a_similarity, dim=(1, 2)) / (a_similarity.shape[1] ** 2)
#     # print(a_norm.shape)
#     a_norm = torch.mean(a_norm)
#     return a_norm


# def regurize(mul, std):
#     variance_dul = std ** 2
#     variance_dul = variance_dul.view(variance_dul.shape[0], -1)
#     mul = mul.view(mul.shape[0], -1)
#     loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
#     loss_kl = torch.mean(loss_kl)

#     return loss_kl


# def get_feature_diff(x1, x2):
#     # print(x1.shape,x2.shape)
#     x1 = F.adaptive_avg_pool2d(x1, (7, 7))
#     x2 = F.adaptive_avg_pool2d(x2, (7, 7))
#     # x1 = torch.mean(x1, dim=(2, 3))
#     # x2 = torch.mean(x2, dim=(2, 3))

#     x1 = x1.permute(0, 2, 3, 1).contiguous()
#     x2 = x2.permute(0, 2, 3, 1).contiguous()

#     rgb = x1.view(-1, x1.shape[3])
#     depth = x2.view(-1, x2.shape[3])

#     diff = F.mse_loss(rgb, depth)
#     # diff = torch.cosine_similarity(rgb, depth)
#     # diff = torch.mean(diff)
#     # print(simi.shape)
#     return diff

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0])
    return x.float().var() / (x.float().mean() ** 2 + eps)


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    print("Start training ... ")

    loss_total = 0
    combi_score_sum = 0
    combi_score_count= 0
    for step, (spec, image, label) in enumerate(dataloader):
        # pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        
        pred_label, pred_label_full, load, importance, gate_distill_loss, var_order_loss, mod_combi, mismatch_vec = model(spec.unsqueeze(1).float(), image.float())


        label_loss = criterion(pred_label, label) + criterion(pred_label_full, label)
        gate_loss = cv_squared(importance) + cv_squared(load)
        loss = args.weight_task * label_loss + args.weight_lb* gate_loss.mean() + args.weight_distill * gate_distill_loss.mean() + args.weight_order* var_order_loss

        loss.backward()
        optimizer.step()

        loss_total += loss.item()

        mod_combi = mod_combi.int()

        codes = (mod_combi[:,0] << 1) + mod_combi[:,1]  # 10→2, 01→1

        pattern_order = torch.tensor([2, 1], device=f'cuda:{args.gpu_ids}')

        batch_score_sums = torch.bincount(codes, weights=mismatch_vec, minlength=3)[pattern_order]
        batch_score_counts = torch.bincount(codes, minlength=4)[pattern_order]

        combi_score_sum += batch_score_sums.detach().cpu().numpy()
        combi_score_count += batch_score_counts.detach().cpu().numpy()

    scheduler.step()
    # import pdb; pdb.set_trace()

    if epoch >= args.prob_epoch:
        new_prob = np.where(combi_score_count != 0, combi_score_sum / combi_score_count, 0.0)
        new_prob = np.exp(new_prob/args.temperature) / np.exp(new_prob/args.temperature).sum()
        update_prob = np.maximum(new_prob, 0.03)
        model.p = update_prob / update_prob.sum()

    return loss_total/len(dataloader)

def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out_ = model(spec.unsqueeze(1).float(), image.float())
            out_ = out_[0]
            out = out_[torch.arange(2,out_.size(0),3)]
            out_v = out_[torch.arange(1,out_.size(0),3)]
            out_a = out_[torch.arange(0,out_.size(0),3)]

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    args.p = [0.5, 0.5]
    args.weak_modality = 'none'

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    if args.train:

        best_acc = 0.0

        std_a_sum = 0
        std_v_sum = 0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                train_loss = train_epoch(args,
                                        epoch,
                                        model,
                                        device,
                                        train_dataloader,
                                        optimizer,
                                        scheduler)
                
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)


                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                train_loss = train_epoch(args,
                                        epoch,
                                        model,
                                        device,
                                        train_dataloader,
                                        optimizer,
                                        scheduler)
                
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)


            if acc > best_acc and epoch:
                best_acc = (float(acc)+float(acc_a)+float(acc_v))/3

                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                # save_dir = os.path.join(args.ckpt_path, f"{args.name}_{acc:.3f}")
                save_dir = os.path.join(args.ckpt_path, args.name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("AVG ACC: {:.4f}, All_Acc: {:.3f}".format((float(acc)+float(acc_a)+float(acc_v))/3, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("AVG ACC: {:.4f}, All_Acc: {:.3f}, Best Acc: {:.3f}".format((float(acc)+float(acc_a)+float(acc_v))/3, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
        # print(state_dict)
        model.load_state_dict(state_dict)
        # model.train()
        # model.eval()
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
