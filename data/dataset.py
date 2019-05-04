import re, os, random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data

class FaceLandmarkDataset(data.Dataset):
    def __init__(self, opt):
        super(FaceLandmarkDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.info_list = _get_info_list(opt)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=opt['mean'], std=[1.0, 1.0, 1.0])
        ])

    def __getitem__(self, index):
        info = self.info_list[index]
        img = Image.open(info[0]).convert('RGB')
        img, landmark, landmark_original = self._resize(img, info)
        gt = _generate_gt((img.size[0] // 4, img.size[1] // 4), landmark,
                               self.opt['gt_sigma'])  # C*H*W
        img, gt = self._random_modify(img, gt)
        img = self.transform(img)
        gt = torch.from_numpy(np.ascontiguousarray(gt))
        return {'img': img, 'heatmap_gt': gt, 'path': info[0], 'landmark_gt': np.array(landmark_original)}

    def __len__(self):
        return len(self.info_list)

    def _resize(self, img, info):
        landmark = info[1]
        if self.opt['type'] == 'celebA':
            width, height = img.size
            new_width, new_height = min(width, height), min(width, height)
            left = (width - new_width)//2
            top = (height - new_height)//2
            right = (width + new_width)//2
            bottom = (height + new_height)//2
        elif self.opt['type'] == '300W':
            width, height = img.size
            left, top, right, bottom = info[2]
            # expand boundding box to 1.2X
            delta_x = int((right - left) * 0.1) 
            delta_y = int((bottom - top) * 0.1) 
            left = max(0, left - delta_x)
            right = min(width, right + delta_x)
            top = max(0, top - delta_y)
            bottom = min(height, bottom + delta_y)
            new_width, new_height = right - left, bottom - top 
        else:
            raise NotImplementedError(
                'Dataset type %s is not recognized' % (opt['type']))
        
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((self.opt['input_length'], self.opt['input_length']), Image.BICUBIC)
        scale_0 = 1.0 * self.opt['input_length'] / new_width
        scale_1 = 1.0 * self.opt['input_length'] / new_height
        landmark_original = [((x[0] - left)*scale_0, (x[1] - top)*scale_1) for x in landmark]
        landmark_resized = [(x[0] * 0.25, x[1] * 0.25) for x in landmark_original]
        return resized, landmark_resized, landmark_original
        
    def _random_modify(self, img, gt):
        if random.random() > 0.5 and self.opt['flip_v']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            gt = np.flip(gt, 1)
        if random.random() > 0.5 and self.opt['flip_h']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = np.flip(gt, 2)
        if self.opt['rotate']:
            rot_rand = random.random()
            if rot_rand > 0.75:
                img = img.transpose(Image.ROTATE_90)
                gt = np.rot90(gt, k=1, axes=(1, 2))
            elif rot_rand > 0.5:
                img = img.transpose(Image.ROTATE_180)
                gt = np.rot90(gt, k=2, axes=(1, 2))
            elif rot_rand > 0.25:
                img = img.transpose(Image.ROTATE_270)
                gt = np.rot90(gt, k=3, axes=(1, 2))
        return img, gt

def _generate_gt(size, landmark_list, sigma):
    heatmap_list = [
        _generate_one_heatmap(size, l, sigma) for l in landmark_list
    ]
    return np.stack(heatmap_list, axis=0)

def _generate_one_heatmap(size, landmark, sigma):
    w, h = size
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap

def _get_info_list(opt):
    info_list = []
    if opt['type'] == 'celebA':
        img_root = opt['image_root']
        with open(opt['annotation_path'], 'r') as f:
            anno_list = f.read().strip('\n').split('\n')[2:]
        with open(opt['partition_path'], 'r') as f:
            part_list = sorted(f.read().strip('\n').split('\n'))
        for i, anno in enumerate(sorted(anno_list)):
            if opt['phase'] == 'train':
                if part_list[i][-1] != '0':
                    continue
            elif opt['phase'] == 'val':
                if part_list[i][-1] != '1':
                    continue
            elif opt['phase'] == 'test':
                if part_list[i][-1] != '2':
                    continue
            else:
                raise NotImplementedError(
                    'Dataset phase %s is not recognized' % (opt['phase']))

            split = re.split('\W+', anno)
            img_name = split[0] + '.' + split[1]
            assert len(split) == opt['num_keypoints'] * 2 + 2
            landmark = []
            for i in range(opt['num_keypoints']):
                landmark.append((int(split[2 * i + 2]), int(split[2 * i + 3])))
            info_list.append((os.path.join(img_root, img_name), landmark))

    elif opt['type'] == '300W':
        # 300W list file generated according to 
        # https://github.com/facebookresearch/supervision-by-registration/tree/master/cache_data
        with open(opt['annotation_path'], 'r') as f:
            anno_list = f.read().splitlines()
        for anno in anno_list:
            split_anno = anno.split(' ')
            img_path = split_anno[0]
            landmark = pts_reader(split_anno[1])
            bbox = [float(x) for x in split_anno[2:]]
            info_list.append((img_path, landmark, bbox))
            
    else:
        raise NotImplementedError(
            'Dataset type %s is not recognized' % (opt['type']))
    if not opt['phase'] == 'train' and len(info_list) > 1000:
        info_list = info_list[:1000]
    return info_list

def pts_reader(pts_path):
    with open(pts_path, 'r') as f:
        rows = [rows.strip() for rows in f]
    head = rows.index('{') + 1
    tail = rows.index('}')
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return points
