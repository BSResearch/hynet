import torch.utils.data

from .base_dataset import collate_fn
from .base_dataset import collate_fn_2


def create_dataset(opt):
    """loads dataset class"""
    if opt.dataset_mode == 'segmentation':
        from .segmentation_data import SegmentationData
        dataset = SegmentationData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = create_dataset(opt)
        if not opt.save_segmentation_for_test_files:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=opt.num_threads,
                collate_fn=collate_fn)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=not opt.serial_batches,
                num_workers=opt.num_threads,
                collate_fn=collate_fn_2)

    def __len__(self):
        return len(self.dataset)
