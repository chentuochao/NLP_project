import numpy as np
import torch
import CLIP.clip as clip
from PIL import Image
import pickle
import os
import os.path as path
from torchvision.datasets import CocoCaptions
import time
import json
from gen_feat_misc import (
    gender_attr, 
    neutralize_gender,
    read_ids,
    read_caps,
    caps_from_anno,
)
from tqdm import tqdm
from mixed_KSG import mixed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', help='path to datasets')
parser.add_argument('--name', default='coco', choices=['coco', 'flickr30k'], help='name of dataset')
parser.add_argument('--out_ext', default='-clip.pkl', help='Extension of output file')
parser.add_argument('--num_train_sample', type=int, default=10000, help='Number of samples in training set.')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU index')
args = parser.parse_args()

DATA_PATH = args.data_path
OUT_FILE = 'karpathy-' + args.name + '_{}'.format(args.num_train_sample)
OUT_FILE += args.out_ext

if 'coco' in args.name:
    train_root = 'COCO/train2014'
    test_root = 'COCO/val2014'
    anno_file = 'dataset_coco.json'
elif 'flickr' in args.name:
    train_root = 'flickr30k-images'
    test_root = 'flickr30k-images'
    anno_file = 'dataset_flickr30k.json'
data_info = json.load(open(anno_file))
train_inds = [ind for ind, x in enumerate(data_info['images']) if 'train' in x['split']]
train_inds = train_inds if args.num_train_sample <= 0 else train_inds[:args.num_train_sample]
train_finames = [x['filename'] for x in np.array(data_info['images'], dtype=object)[train_inds]]
train_finames = [path.join(DATA_PATH, train_root, x) for x in train_finames]
assert all([path.isfile(x) for x in train_finames]), 'Some image files does not exist in {}-set.'.format('train')
test_inds = [ind for ind, x in enumerate(data_info['images']) if 'test' in x['split']]
test_finames = [x['filename'] for x in np.array(data_info['images'], dtype=object)[test_inds]]
test_finames = [path.join(DATA_PATH, test_root, x) for x in test_finames]
assert all([path.isfile(x) for x in test_finames]), 'Some image files does not exist in {}-set.'.format('test')

word_bank_fis = dict(
    male='male_word_bank.txt',
    female='female_word_bank.txt',
    neutral='neutral_word_bank.txt',
)
gender_dict = dict(male=0, neutral=1, female=2)
begin = time.time()

device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

train_caps = caps_from_anno(data_info, train_inds, device, repeat=5)
test_caps = caps_from_anno(data_info, test_inds, device, repeat=5)

# [Train] Generate gender label
train_gender = gender_attr(train_caps, word_bank_fis, size_parti=5)
train_gender = np.array(train_gender)

# [Train] Load Image and get image features
remain_ids = [*range(len(train_finames))]
train_ifeats = []
with torch.no_grad():
    for idx, img_fi in tqdm(enumerate(train_finames)):
        try:
            image = Image.open(img_fi).convert('RGB')
            inp_ft = transform(image).to(device)[None, ...]
            img_ft = model.encode_image(inp_ft).float()
            train_ifeats.append(img_ft)
        except:
            print("File {} does not exist.".format(img_fi))
            remain_ids.remove(idx)
    train_ifeats = torch.cat(train_ifeats).cpu().numpy()

# [Train] Get Mutual Informations
mis = []
ft_dim = train_ifeats.shape[1]
for ft_id in tqdm(range(ft_dim)):
    np_feat = train_ifeats[:,ft_id].squeeze()[..., None]
    mis.append(mixed.Mixed_KSG(np_feat, train_gender[remain_ids]))
mis = np.array(mis)

# [Test] Generate gender label
test_gender = gender_attr(test_caps, word_bank_fis, size_parti=5)
test_gender = np.array(test_gender)
test_caps = test_caps[::5]
test_caps = neutralize_gender(test_caps, word_bank_fis)

# [Test] Get text features
test_tfeats = []
test_tin = torch.zeros(1, model.context_length, dtype=torch.long)
with torch.no_grad():
    for i, caption in tqdm(enumerate(test_caps)):
        test_tin[0, :len(caption[0])] = caption[0]
        tfeat = model.encode_text(test_tin.to(device)).float()
        test_tfeats.append(tfeat)
        test_tin.zero_()
    test_tfeats = torch.cat(test_tfeats).cpu().numpy()

# [Test] Load Image and get image features
test_ifeats = []
with torch.no_grad():
    for img_fi in tqdm(test_finames):
        image = Image.open(img_fi).convert('RGB')
        inp_ft = transform(image).to(device)[None, ...]
        img_ft = model.encode_image(inp_ft).float()
        test_ifeats.append(img_ft)
    test_ifeats = torch.cat(test_ifeats).cpu().numpy()

# Store output
output = dict()
output['image_feat'] = test_ifeats
output['text_feat'] = test_tfeats
output['gender'] = test_gender
output['mu_info'] = mis
output['gender2cateid'] = dict(male=0, neutral=1, female=2)

# Save output information
with open(OUT_FILE, 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
