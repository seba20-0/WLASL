import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import videotransforms

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)
parser.add_argument('-weights', type=str, help='path to trained model weights')

args = parser.parse_args()


def run(configs, mode='rgb', root='/ssd/Charades_v1_rgb',
             test_split='',
             weights=None):

    print("Testing")
    print(configs)

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_dataset = Dataset(test_split, 'test', root, mode, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                            num_workers=2, pin_memory=False)

    num_classes = test_dataset.num_classes

    # setup model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)

    if weights:
        print(f"Loading model weights from {weights}")
        i3d.load_state_dict(torch.load(weights))

    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    total_loc_loss = 0.0
    total_cls_loss = 0.0
    num_iter = 0

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    start_time = time.time()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            if data == -1:
                continue

            inputs, labels, vid = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            t = inputs.size(2)

            per_frame_logits = i3d(inputs)
            per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

            # localization and classification loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            cls_loss = F.binary_cross_entropy_with_logits(
                torch.max(per_frame_logits, dim=2)[0],
                torch.max(labels, dim=2)[0]
            )

            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            num_iter += 1

            predictions = torch.max(per_frame_logits, dim=2)[0]
            gt = torch.max(labels, dim=2)[0]

            for i in range(per_frame_logits.shape[0]):
                confusion_matrix[
                    torch.argmax(gt[i]).item(),
                    torch.argmax(predictions[i]).item()
                ] += 1

    test_acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
    end_time = time.time()

    print("\n=== Testing Results ===")
    print(f"Localization Loss: {total_loc_loss / num_iter:.4f}")
    print(f"Classification Loss: {total_cls_loss / num_iter:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")

    total_time = end_time - start_time
    hrs = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = int(total_time % 60)
    print(f"Inference completed in {hrs}h {mins}m {secs}s")


if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': 'WLASL/videos'}
    test_split = 'WLASL/nslt_leaveout_user11_updated.json'
    weights = 'WLASL/checkpoints_user1/nslt_10_epoch10.pt'
    config_file = 'WLASL/code/I3D/configfiles/asl2000.ini'

    configs = Config(config_file)
    run(configs=configs, mode=mode, root=root, test_split=test_split, weights=weights)
