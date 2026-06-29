from surf_txt import SURF, surf_multi_transforms_test
import torch
import torchvision.transforms as tt
from tqdm import tqdm
import numpy as np
import csv

def calc_ACER(pred_label, target_label):
    living_right = ((pred_label==target_label)&(target_label==1)).sum()
    living_wrong = ((pred_label!=target_label)&(target_label==1)).sum()
    spoofing_right = ((pred_label==target_label)&(target_label==0)).sum()
    spoofing_wrong = ((pred_label!=target_label)&(target_label==0)).sum()

    APCER = living_wrong / (living_wrong + living_right)
    NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)
    ACER = (APCER + NPCER) / 2
    return ACER

def calc_ACER_multi(model, loader, args, verbose=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    # model.train(False)
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to(f'cuda:{args.gpu}')
    outputs_full = []
    labels_full = []

    with torch.no_grad():

        for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample['binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.to(f'cuda:{args.gpu}')
                img_ir = img_ir.to(f'cuda:{args.gpu}')
                img_depth = img_depth.to(f'cuda:{args.gpu}')
                target = target.to(f'cuda:{args.gpu}')

                outputs_batchs = model(img_rgb, img_ir, img_depth)
                if isinstance(outputs_batchs, tuple):
                    outputs_batch = outputs_batchs[0]
                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(target)


        model.train(mode_saved)
        outputs_full = torch.cat(outputs_full, dim=0)
        labels_full = torch.cat(labels_full, dim=0)

        _, labels_predicted = torch.max(outputs_full.data, dim=1)

        pred_label_combi0 = labels_predicted[torch.arange(0,labels_predicted.size(0),7)]
        pred_label_combi1 = labels_predicted[torch.arange(1,labels_predicted.size(0),7)]
        pred_label_combi2 = labels_predicted[torch.arange(2,labels_predicted.size(0),7)]
        pred_label_combi3 = labels_predicted[torch.arange(3,labels_predicted.size(0),7)]
        pred_label_combi4 = labels_predicted[torch.arange(4,labels_predicted.size(0),7)]
        pred_label_combi5 = labels_predicted[torch.arange(5,labels_predicted.size(0),7)]
        pred_label_combi6 = labels_predicted[torch.arange(6,labels_predicted.size(0),7)]

        ACER_combi0 = calc_ACER(pred_label_combi0, labels_full)
        ACER_combi1 = calc_ACER(pred_label_combi1, labels_full)
        ACER_combi2 = calc_ACER(pred_label_combi2, labels_full)
        ACER_combi3 = calc_ACER(pred_label_combi3, labels_full)
        ACER_combi4 = calc_ACER(pred_label_combi4, labels_full)
        ACER_combi5 = calc_ACER(pred_label_combi5, labels_full)
        ACER_combi6 = calc_ACER(pred_label_combi6, labels_full)

    return [ACER_combi0, ACER_combi1, ACER_combi2, ACER_combi3, ACER_combi4, ACER_combi5, ACER_combi6]


def batch_test(model, args):

    modality_combination = ['RGB','IR','Depth','RGB+IR','RGB+Depth','IR+Depth','RGB+IR+Depth']

    root_dir = "../data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8)

    results = calc_ACER_multi(model=model, loader=test_loader, args=args, verbose=True)

    log_dir = args.log_root + '/' + args.name + '.csv'
    
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        for idx in [0,2,1,4,3,5,6]:
            my_writer.writerow([f'{modality_combination[idx]:<12} : {results[idx]:.4f}'])


    return sum(results)/len(results)