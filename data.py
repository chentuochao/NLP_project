# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod

def Neutralize(words_list):
    man_words =   ['man','men','boy', 'boys', 'gentleman', 'gentlemen', 'father', 'fathers', 'dad', 'dads', 'son', 'sons', 'male', 'husband', 'husbands', 'boyfriend', 'boyfriends','brother', 'brothers',] 
    #man_words = ['man',   'men',   'boy',   'boys', 'gentleman',  'father', 'dad', 'son', 'male', 'husband', 'boyfriend', 'brother']
    woman_words =   ['woman', 'women', 'girl', 'girls', 'lady', 'ladies', 'mother', 'mothers', 'mom', 'moms','daughter','daughters', 'female', 'wife', 'wives', 'girlfriend', 'girlfriends', 'sister','sisters']
    #woman_words = ['woman', 'women', 'girl', 'girls',      'lady',  'mother', 'mom', 'daughter', 'female', 'wife', 'girlfriend', 'sister']
    replace_words = ['person', 'people', 'child', 'children','adult', 'adults', 'parent', 'parents','parent', 'parents', 'child', 'children', '', 'spouse', 'spouse', 'partner', 'partners','sibling', 'sibling']
    #replace_words = ['person', 'people', 'child', 'children','adult', 'parent', 'parent', 'child', '', 'spouse', 'partner', 'sibling', 'sibling']
    
    assert len(replace_words) == len(woman_words)
    assert len(man_words) == len(woman_words)
    new_sent = []
    if_replace = False
    
    for i in range(0, len(words_list)):
        w = words_list[i]
        if w in man_words:
            idx = man_words.index(w)
            replace_w = replace_words[idx]
            if_replace = True
        elif w in woman_words:
            idx = woman_words.index(w)
            replace_w = replace_words[idx]
            if_replace = True
        else:
            replace_w = w
        
        new_sent.append(replace_w)
        '''
        if i < len(words_list) -1:
            new_sent = new_sent + replace_w + ' '
        else:
            new_sent = new_sent + replace_w 
        '''
    return if_replace, new_sent    

def contain(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return True
    return False

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, if_Neutralize):
        self.vocab = vocab
        self.if_Neutralize = if_Neutralize
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # male and female word bank
        self.MALE_WORD_BANK = ['man','men', 'male', 'males', 'boy', 'boys', 'gentleman', 'gentlemen', 'father', 'fathers', 'dad', 'dads','brother', 'brothers', 'son', 'sons', 'husband', 'husbands', 'boyfriend', 'boyfriends'] #["man","men", "male","boy","father","brother","son","husband","boyfriend","gentleman"]

        self.FEMALE_WORD_BANK = ['woman', 'women', 'female', 'females', 'girl', 'girls', 'lady', 'ladies', 'mother', 'mothers', 'mom', 'moms', 'sister','sisters','daughter','daughters','wife', 'wives', 'girlfriend', 'girlfriends'] #["woman", "women","female","girl","lady","mother","mom","sister","daughter","wife","girlfriend"]

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split, mmap_mode = 'r')
        self.length = len(self.captions)
        print(self.images.shape, self.length )
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab
        #print(index, caption)
        # Convert caption (string) to word ids.
        cap = str(caption, 'utf-8').lower()
        tokens = nltk.tokenize.word_tokenize(cap)
        gender = 0
        old_tokens = tokens
        have_man = contain(tokens, self.MALE_WORD_BANK)
        have_woman = contain(tokens, self.FEMALE_WORD_BANK)
        if have_man and not have_woman:
            gender = 1
        elif have_woman and not have_man:
            gender = 2
        elif have_woman and have_man:
            gender = 3
        else:
            gender = 0

        if self.if_Neutralize:
            if_replace, tokens = Neutralize(tokens)
            
            #if if_replace:
            #    print(have_man,have_woman,gender)
            #    print(old_tokens)
            #    print(tokens)
            
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        #print(target)
        return image, target, index, img_id, gender

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, genders = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids, genders


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab, opt.neutralize)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
