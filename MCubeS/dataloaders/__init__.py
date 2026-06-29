from dataloaders.datasets import multimodal_dataset
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    train_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='train')
    val_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='val')
    test_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='test')

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, num_class

