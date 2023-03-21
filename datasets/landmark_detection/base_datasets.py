import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

def generate_target(joints, joints_vis, heatmap_size, sigma, image_size):
    """Generate heatamap for joints.

    Args:
        joints: (K, 2)
        joints_vis: (K, 1)
        heatmap_size: W, H
        sigma:
        image_size:

    Returns:

    """
    num_joints = joints.shape[0]
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3
    image_size = np.array(image_size)
    heatmap_size = np.array(heatmap_size)

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if mu_x >= heatmap_size[0] or mu_y >= heatmap_size[1] \
                or mu_x < 0 or mu_y < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight
    
class BaseTrainDataset(Dataset):

    def __init__(self, examples, transform, heatmap_size=None, image_size=None, landmarks_count=68):
        # examples is a list of [img_id(path), target_landmarks(list[int]), group(int)]
        # transform is a torchvision.transforms.Compose(...)
        self.examples = examples
        self.transform_lm, self.transform = transform
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.landmarks_count = landmarks_count

        group0 = []
        group1 = []

        for example in examples:
            if example[-1] == 0:
                group0.append(example)
            else:
                group1.append(example)

        self.dataset_len = max(len(group0), len(group1))

        if len(group0) > len(group1):
            self.source = group0
            self.target = group1
        else:
            self.source = group1
            self.target = group0

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        s_id, s_targ, _ = self.source[index]
        t_id, t_targ, _ = random.choice(self.target)

        s_img = Image.open(s_id).convert('RGB')
        t_img = Image.open(t_id).convert('RGB')
        
        s_img, s_targ = self.transform_lm(np.array(s_img, dtype=np.uint8), np.array(s_targ).reshape(-1, 2))
        t_img, t_targ = self.transform_lm(np.array(t_img, dtype=np.uint8), np.array(t_targ).reshape(-1, 2))

        s_img = self.transform(Image.fromarray(s_img))
        t_img = self.transform(Image.fromarray(t_img))

        visible = np.ones((self.landmarks_count, ), dtype=np.float32)
        visible = visible[:, np.newaxis]

        # 2D Heatmap
        s_targ, s_targ_weight = generate_target(s_targ, visible, self.heatmap_size, 2, self.image_size)
        s_targ = torch.from_numpy(s_targ)
        s_targ_weight = torch.from_numpy(s_targ_weight)

        t_targ, t_targ_weight = generate_target(t_targ, visible, self.heatmap_size, 2, self.image_size)
        t_targ = torch.from_numpy(t_targ)
        t_targ_weight = torch.from_numpy(t_targ_weight)

        return s_img, s_targ, s_targ_weight, t_img, t_targ, t_targ_weight
    
class BaseTestDataset(Dataset):

    def __init__(self, examples, transform, heatmap_size=None, image_size=None, landmarks_count=68):
        # examples is a list of [img_id(path), target_landmarks(list[int]), group(int)]
        # transform is a torchvision.transforms.Compose(...)
        self.examples = examples
        self.transform_lm, self.transform = transform
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.landmarks_count = landmarks_count

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        id, targ, group = self.examples[index]

        img = Image.open(id).convert('RGB')
        
        img, targ = self.transform_lm(np.array(img, dtype=np.uint8), np.array(targ).reshape(-1, 2))

        img = self.transform(Image.fromarray(img))

        visible = np.ones((self.landmarks_count, ), dtype=np.float32)
        visible = visible[:, np.newaxis]

        # 2D Heatmap
        targ, targ_weight = generate_target(targ, visible, self.heatmap_size, 2, self.image_size)
        targ = torch.from_numpy(targ)
        targ_weight = torch.from_numpy(targ_weight)

        return img, targ, targ_weight, group
