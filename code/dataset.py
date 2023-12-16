import os
from torch.utils.data import Dataset
from osgeo import gdal


class GeoCropDataset(Dataset):
    def __init__(self, root='data/', is_train=True, transform=None):
        super().__init__()
        file_name = 'training_data.txt' if is_train else 'validation_data.txt'
        with open(os.path.join(root, file_name)) as f:
            chip_list = [line.rstrip() for line in f]
        self.img_list = [os.path.join(root, 'hls', chip + '_merged.tif') for chip in chip_list]
        self.mask_list = [os.path.join(root, 'masks', chip + '.mask.tif') for chip in chip_list]
        self.transform = transform
        

    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        imgs = gdal.Open(img_path, gdal.GA_ReadOnly).ReadAsArray()
        mask = gdal.Open(mask_path, gdal.GA_ReadOnly).ReadAsArray()
    
        assert imgs.shape == (18, 224, 224)
        assert mask.shape == (224, 224)

        if self.transform:
            imgs, mask = self.transform(imgs, mask)

        return imgs, mask
    
