from models.our_model import OURMODEL
from argparse import ArgumentParser
from test_func import batch_test
from utils import surf_baseline_multi_dataloader
from train_func import train_model_multi

import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = ArgumentParser()
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--se_reduction', type=int, default=16, help='para for se layer')
parser.add_argument('--modal', type=str, default='multi')
parser.add_argument('--inplace_new', type=int, default=384, help='para for se layer')
parser.add_argument('--data_root', type=str,
                    default='../data/CASIA-SURF')
parser.add_argument('--miss_modal', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=5)
parser.add_argument('--num_experts', type=int, default=16)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--name', type=str, default='ours_default')
parser.add_argument('--model_root', type=str, default='./output/models')
parser.add_argument('--log_root', type=str, default='./output/logs')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--anchor_task', type=float, default=1)
parser.add_argument('--sampled_task', type=float, default=1)
parser.add_argument('--load_balance', type=float, default=1)
parser.add_argument('--gate_distill', type=float, default=1)
parser.add_argument('--residual_reconst', type=float, default=1)
parser.add_argument('--var_order', type=float, default=1)

parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--prob_epoch', type=int, default=101)
parser.add_argument('--temperature', type=float, default=1.)




args = parser.parse_args()
args.log_name = args.name + '.csv'
args.model_name = args.name

seed_torch()
model = OURMODEL(args).to(f'cuda:{args.gpu}')
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

train_loader = surf_baseline_multi_dataloader(train=True, args=args)

root_dir = "../data/CASIA-SURF"
from surf_txt import SURF, surf_multi_transforms_test
txt_dir = root_dir + '/test_private_list.txt'
surf_dataset = SURF(txt_dir=txt_dir,
                    root_dir=root_dir,
                    transform=surf_multi_transforms_test, miss_modal=args.miss_modal)

test_loader = torch.utils.data.DataLoader(
    dataset=surf_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8)

train_model_multi(model, criterion, optimizer, train_loader, test_loader, args)