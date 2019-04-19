from .dataset import FaceLandmarkDataset
from torch.utils.data import DataLoader

def create_dataloader(opt):
    ds = FaceLandmarkDataset(opt)
    shuffle = opt['phase'] == 'train'
    dl = DataLoader(ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=shuffle)
    return dl