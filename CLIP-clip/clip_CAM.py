import numpy as np
import pickle
import os
import os.path as path
import time
import argparse
import eval_tools as EVAL
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import CLIP.clip as clip
import torch
from pytorch_grad_cam import (
    GradCAM, 
    ScoreCAM, 
    GradCAMPlusPlus, 
    AblationCAM, 
    XGradCAM, 
    EigenCAM, 
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from gen_feat_misc import (
    gender_attr, 
    neutralize_gender,
    read_ids,
    read_caps,
)
import json
from pycocotools.coco import COCO
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0, type=int, help='GPU index')
parser.add_argument('--data_path', default='data', help='path to datasets')
parser.add_argument('--name', default='coco', choices=['coco', 'flickr30k'], help='name of dataset')
parser.add_argument('--anno_file', default='dataset_coco.json', help='Filename of annotation file.')
parser.add_argument('--anno_seg_file', default='instances_val2014.json', help='Filename of annotation file.')
parser.add_argument('--cam_dim', type=int, default=10, help='Number of dimension to visualize.')
parser.add_argument('--target_layer', type=int, default=-1, help='Target layer of to visualize gradient.')
parser.add_argument('--num_cat', type=int, default=20, help='Plot the first num_cat categories with highest activation.')
args = parser.parse_args()


device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

with open('karpathy-coco_10000-clip.pkl', 'rb') as handle:
    output = pickle.load(handle)
    
image_features = output['image_feat']
text_features = output['text_feat']
gender = output['gender']
mis = output['mu_info']
gender2catid = output['gender2cateid']
ft_dim = image_features.shape[1]
num_sample = text_features.shape[0]


test_root = 'COCO/val2014'
data_info = json.load(open(args.anno_file))
coco = COCO(path.join(args.data_path, 'COCO/annotations', args.anno_seg_file))
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)
filterClasses = ['person'] # ['laptop', 'tv', 'cell phone']
catIds = coco.getCatIds(catNms=filterClasses) 
cat2id = {x['name']: x['id'] for x in cats}
id2cat = {x['id']: x['name'] for x in cats}
cat2area = dict()
cat2area = {'person': 32556352.0, 'bicycle': 687236.0, 'car': 3563176.0, 'motorcycle': 1159201.0, 'airplane': 621337.0, 'bus': 1239838.0, 'train': 1814778.0, 'truck': 1512631.0, 'boat': 1029789.0, 'traffic light': 1096258.0, 'fire hydrant': 263226.0, 'stop sign': 328646.0, 'parking meter': 388386.0, 'bench': 1169344.0, 'bird': 628647.0, 'cat': 1946895.0, 'dog': 1185943.0, 'horse': 1104545.0, 'sheep': 1294074.0, 'cow': 1031637.0, 'elephant': 870988.0, 'bear': 365490.0, 'zebra': 1056045.0, 'giraffe': 799319.0, 'backpack': 993046.0, 'umbrella': 998126.0, 'handbag': 1656549.0, 'tie': 695274.0, 'suitcase': 1292549.0, 'frisbee': 376128.0, 'skis': 676800.0, 'snowboard': 259471.0, 'sports ball': 253013.0, 'kite': 550665.0, 'baseball bat': 265361.0, 'baseball glove': 382913.0, 'skateboard': 697270.0, 'surfboard': 668382.0, 'tennis racket': 488032.0, 'bottle': 5705052.0, 'wine glass': 1485370.0, 'cup': 4375084.0, 'fork': 1680300.0, 'knife': 1819638.0, 'spoon': 2004655.0, 'bowl': 3231138.0, 'banana': 2135422.0, 'apple': 947676.0, 'sandwich': 1517839.0, 'orange': 1492943.0, 'broccoli': 4311543.0, 'carrot': 4255121.0, 'hot dog': 597333.0, 'pizza': 1956410.0, 'donut': 1529482.0, 'cake': 1416348.0, 'chair': 3911683.0, 'couch': 1121823.0, 'potted plant': 1049125.0, 'bed': 1824408.0, 'dining table': 4117286.0, 'toilet': 908192.0, 'tv': 830993.0, 'laptop': 876088.0, 'mouse': 404410.0, 'remote': 1543473.0, 'keyboard': 622564.0, 'cell phone': 862504.0, 'microwave': 241760.0, 'oven': 523809.0, 'toaster': 47476.0, 'sink': 646362.0, 'refrigerator': 813518.0, 'book': 4686148.0, 'clock': 719668.0, 'vase': 868419.0, 'scissors': 382340.0, 'teddy bear': 1389437.0, 'hair drier': 59357.0, 'toothbrush': 357853.0}
imgIds = coco.getImgIds(catIds=catIds)

test_inds = [ind for ind, x in enumerate(data_info['images']) if 'test' in x['split']]
test_finames = [x['filename'] for x in np.array(data_info['images'], dtype=object)[test_inds]]
test_finames = [path.join(args.data_path, test_root, x) for x in test_finames]
test_cocoids = [x['cocoid'] for x in np.array(data_info['images'], dtype=object)[test_inds]]

if len(cat2area) == 0:    
    for cocoid, fname in tqdm(zip(test_cocoids, test_finames)):
        annIds = coco.getAnnIds(imgIds=cocoid, iscrowd=False) # , catIds=catIds, iscrowd=None)
        img = Image.open(fname)
        width, height = img.size
        inp_ft = transform(img).to(device)[None, ...]
        resize_height, resize_width = inp_ft.shape[-2:]
        resize_factor = (resize_height * resize_width) / (width * height)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            mask = coco.annToMask(anns)
            mask = cv2.resize(mask, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
            cat_name = id2cat[ann['category_id']]
            cat2area[cat_name] = cat2area.get(cat_name, 0) + np.sum(mask)

def reshape_transform(tensor):
    result = tensor.permute(1, 0, 2)
    result = result[:, 1:, :].reshape(
        result.size(0), 
        int((result.size(1) - 1)**0.5),
        int((result.size(1) - 1)**0.5),
        result.size(2),
    )
    result = result.transpose(2, 3).transpose(1, 2)
    
    return result
acti_by_cat = dict()
with torch.cuda.device(device):
    sorted_dims = np.argsort(mis)[::-1]
    for cocoid, fname in tqdm(zip(test_cocoids, test_finames)):
        acti_maps = []
        img = Image.open(fname).convert('RGB')
        width, height = img.size
        inp_ft = transform(img).to(device)[None, ...].type(torch.float16)
        resize_height, resize_width = inp_ft.shape[-2:]
        ann_masks = dict()
        annIds = coco.getAnnIds(imgIds=cocoid, iscrowd=False)
        anns = coco.loadAnns(annIds)            
        for sid, sdim in enumerate(sorted_dims[:args.cam_dim]):
            cam = GradCAM(
                model=model.visual, 
                target_layers=[model.visual.transformer.resblocks[args.target_layer].ln_1], 
                use_cuda=True,
                reshape_transform=reshape_transform,
            )
            targets = [ClassifierOutputTarget(sdim)]
            grayscale_cam = cam(input_tensor=inp_ft, targets=targets)
            acti_maps.append(grayscale_cam.squeeze())
        for ann in anns:
            cat_name = id2cat[ann['category_id']]
            mask = coco.annToMask(ann)
            mask = cv2.resize(mask, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
            ann_masks[cat_name] = ann_masks[cat_name] + mask if cat_name in ann_masks else mask
            ann_masks[cat_name] = np.clip(ann_masks[cat_name], 0, 1)
        for acti_map in acti_maps:
            for cat_name in ann_masks: 
                acti_by_cat[cat_name] = acti_by_cat.get(cat_name, 0) + (acti_map * ann_masks[cat_name]).sum()
acti_areas = sorted(acti_by_cat.items(), key=lambda x: x[1], reverse=True)
plt.bar([x[0] for x in acti_areas[:args.num_cat]],
        [x[1] for x in acti_areas[:args.num_cat]], 
        width=0.5, 
        bottom=None, 
        align='center',
)
plt.xticks(rotation='vertical')
plt.ylabel('Accumulated Activation')
plt.tight_layout()
plt.savefig('acti-layer-{}.png'.format(args.target_layer))