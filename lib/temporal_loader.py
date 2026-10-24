import torch.nn.functional as F
from torch.utils.data import Dataset

from lib.human_loader import StereoHumanDataset


class ClipDataset(Dataset):
    """T consecutive frames of one source pair; sample names are <scene>_<set>_<frame:04d>"""

    def __init__(self, opt, clip_len, phase='train'):
        self.base = StereoHumanDataset(opt, phase=phase)
        self.opt, self.T = opt, clip_len
        self.index = {n: i for i, n in enumerate(self.base.sample_list)}
        self.starts = []
        for n in self.base.sample_list:
            prefix, t = n[:-5], int(n[-4:])
            if all('%s_%04d' % (prefix, t + k) in self.index for k in range(clip_len)):
                self.starts.append((prefix, t))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        prefix, t = self.starts[i]
        return [self.base.get_item(self.index['%s_%04d' % (prefix, t + k)], novel_id=self.opt.train_novel_id)
                for k in range(self.T)]


def change_target(img, img_prev, cell=8, thr=8 / 255):
    """changed cells of the feature map: mean colour change of the cell (img in [-1, 1]) above thr"""
    diff = F.avg_pool2d((img - img_prev).abs().mean(1, keepdim=True) / 2, cell)
    return (diff > thr).float()
