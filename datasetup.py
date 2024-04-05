import os
from scantree import scantree, RecursionFilter
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
class CT2MRDataset(Dataset):
    def __init__(self, root, transformations=None, mode='train'):
        self.transform = transforms.Compose(transformations)
        ct_source = scantree(os.path.join(root, '%s/CT' % mode), RecursionFilter(match=['*.png', '*.jpg']))
        self.ct_files =  sorted([path.real for path in ct_source.filepaths()])
        mr_source = scantree(os.path.join(root, '%s/MR' % mode), RecursionFilter(match=['*.png', '*.jpg']))
        self.mr_files =  sorted([path.real for path in mr_source.filepaths()])

    def __getitem__(self, index):
        ct_batch = self.transform(Image.open(self.ct_files[index % len(self.ct_files)]))
        mr_batch = self.transform(Image.open(self.mr_files[index % len(self.mr_files)]))
        return {'CT': ct_batch, 'MR': mr_batch}

    def __len__(self):
        return max(len(self.ct_files), len(self.mr_files))