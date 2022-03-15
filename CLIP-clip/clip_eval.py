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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--name', 
    default='coco', 
    help='Name of exp.'
)
parser.add_argument(
    '--dataset_type', 
    default='coco5k', 
    choices=['coco5k', 'coco1k', 'flickr30k'],
    help='Filename of prediction'
)
parser.add_argument(
    '--num_clip',
    default='0',
    help='Number of dimension to discard',
)
parser.add_argument(
    '--topK',
    type=str,
    default='10',
    help='Number of top-K'
)
parser.add_argument(
    '--plot',
    action='store_true',
    help='To store plot or not.',
)
parser.add_argument(
    '--print',
    action='store_true',
    help='To print result or not.',
)
args = parser.parse_args()

num_clips = [int(x) for x in args.num_clip.split(',')]
topKs = [int(x) for x in args.topK.split(',')]

with open('karpathy-{}_10000-clip.pkl'.format(args.name), 'rb') as handle:
    output = pickle.load(handle)

image_features = output['image_feat']
text_features = output['text_feat']
gender = output['gender']
mis = output['mu_info']
gender2catid = output['gender2cateid']
ft_dim = image_features.shape[1]
num_sample = text_features.shape[0]

biases = {K: dict(mean=[], std=[]) for K in topKs}
recalls = {K: dict(mean=[], std=[]) for K in topKs}

for K in tqdm(topKs):
    for num_clip in num_clips:
        if num_clip > 0:
            rm_dims = np.argpartition(mis, -num_clip)[-num_clip:]
        else:
            if mis.ndim == 1:
                rm_dims = []
            else:
                rm_dims = [[] for _ in range(5)]

        if 'coco1k' in args.dataset_type:
            bias = []
            recall = []
            for i in range(5):
                preserve_dims = sorted(list(set(range(ft_dim)) - set(rm_dims)))
                clip_image_features = image_features[:, preserve_dims]
                clip_text_features = text_features[:, preserve_dims]  
                samp_ids = [*range(i * 1000, (i + 1) * 1000)]
                in_text_features = clip_text_features[samp_ids]
                in_image_features = clip_image_features[samp_ids]
                in_gender = gender[samp_ids]
                sim_map = EVAL.cosine_sims(in_text_features, in_image_features)
                topK_ids = EVAL.pick_topK(sim_map, K)
                bias += [EVAL.eval_gender_biases(topK_ids, in_gender, gender2catid)]
                recall += [EVAL.eval_recalls(topK_ids, np.arange(1000))]
        else:
            preserve_dims = sorted(list(set(range(ft_dim)) - set(rm_dims)))
            clip_image_features = image_features[:, preserve_dims]
            clip_text_features = text_features[:, preserve_dims]        
            sim_map = EVAL.cosine_sims(clip_text_features, clip_image_features)
            topK_ids = EVAL.pick_topK(sim_map, K)
            bias = EVAL.eval_gender_biases(topK_ids, gender, gender2catid)
            recall = EVAL.eval_recalls(topK_ids, np.arange(num_sample))
        if isinstance(bias, list):
            biases[K]['mean'].append(np.array(bias).mean())
            biases[K]['std'].append(np.array(bias).std())
        else:
            biases[K]['mean'].append(bias)
            biases[K]['std'].append(0)
        if isinstance(recall, list):
            recalls[K]['mean'].append(np.array(recall).mean())
            recalls[K]['std'].append(np.array(recall).std())
        else:
            recalls[K]['mean'].append(recall)
            recalls[K]['std'].append(0)

if args.plot:
    plot_data = dict(recall=recalls, bias=biases)
    colors = ['blue', 'orange', 'green']
    labels = ['top-{}'.format(k) for k in topKs]
    ylims = dict(recall=(0, 1), bias=(-0.1, 0.7))
    fig, ax = plt.subplots(1, len(plot_data.keys()))
    for pltidx, data_type in enumerate(plot_data.keys()):
        for curve_id, K in enumerate(topKs):
            mean = np.array(plot_data[data_type][K]['mean'])
            std = np.array(plot_data[data_type][K]['std'])
            ax[pltidx].plot(
                num_clips, 
                mean, 
                colors[curve_id], 
                label=labels[curve_id],
            )
            ax[pltidx].fill_between(
                num_clips, 
                mean - std,
                mean + std,
                color=colors[curve_id], 
                alpha=0.2,
            )

            ax[pltidx].legend()
            ax[pltidx].set_xlabel('clipped dimensions')
            ax[pltidx].set_ylabel(data_type)
            ax[pltidx].set_ylim(ylims[data_type])
            ax[pltidx].set_title(data_type)
    plt.tight_layout()
    plt.savefig('figure-3-{}.png'.format(path.splitext(args.name)[0]))
if args.print:
    for K in topKs:
        print("Top-{}".format(K))
        print("Num clip", end=':\t')
        for idx, num_clip in enumerate(num_clips):
            print("{}".format(num_clip), end=', ')
        print('')
        print("Recall Mean", end=':\t')
        for idx, value in enumerate(recalls[K]['mean']):
            print("{:.4f}".format(value), end=', ')
        print('')
        print("Recall Std", end=':\t')
        for idx, value in enumerate(recalls[K]['std']):
            print("{:.4f}".format(value), end=', ')
        print('')
        print("Bias Mean", end=':\t')
        for idx, value in enumerate(biases[K]['mean']):
            print("{:.4f}".format(value), end=', ')
        print('')
        print("Bias Std ", end=':\t')
        for idx, value in enumerate(biases[K]['std']):
            print("{:.4f}".format(value), end=', ')
        print('')


