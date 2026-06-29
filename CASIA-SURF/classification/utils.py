from surf_txt import SURF, surf_multi_transforms_train, surf_multi_transforms_test
import torch


def surf_baseline_multi_dataloader(train, args):
    # dataset and data loader
    if train:
        txt_dir = args.data_root + '/train_list.txt'
        root_dir = args.data_root

        surf_dataset = SURF(txt_dir=txt_dir,
                            root_dir=root_dir,
                            transform=surf_multi_transforms_train, miss_modal=args.miss_modal)

        surf_data_loader = torch.utils.data.DataLoader(
            dataset=surf_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

    else:
        txt_dir = args.data_root + '/val_private_list.txt'
        root_dir = args.data_root

        surf_dataset = SURF(txt_dir=txt_dir,
                            root_dir=root_dir,
                            transform=surf_multi_transforms_test, miss_modal=args.miss_modal, times=1)

        surf_data_loader = torch.utils.data.DataLoader(
            dataset=surf_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last = True

        )

    return surf_data_loader