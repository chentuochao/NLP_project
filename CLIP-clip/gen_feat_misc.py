import numpy as np
from enum import Enum
import CLIP.clip as clip
import torch
from tqdm import tqdm

class Gender(Enum):
    Male = 0
    Female = 2
    Neutral = 1

def read_ids(fi, max_sample=0, repeat=1):
    print("Reading id from {}".format(fi))
    ids = []
    with open(fi, 'r') as f:
        lines = f.readlines()
        max_sample = max_sample * repeat if max_sample > 0 else len(lines)
        lines = lines[:max_sample]
        for line in tqdm(lines):
            i = int(line.strip())
            if i not in ids:
                ids.append(i)

    return ids

def caps_from_anno(data_info, select_inds, device, repeat=5):
    captions = []
    rawcaps = []
    infos = np.array(data_info['images'], dtype=object)[select_inds]
    for x in tqdm(infos):
        sents = x['sentences']
        for sent in sents[:repeat]:
            rawcaps.append(' '.join(sent['tokens']))
    for cap in tqdm(rawcaps):
        captions.append(clip.tokenize(cap, truncate=True).to(device))
    
    return captions

def read_caps(fi, device, max_sample=0, repeat=5):
    print("Reading caption from {}".format(fi))
    captions = []
    with open(fi, 'r') as f:
        lines = f.readlines()
        max_sample = max_sample * repeat if max_sample > 0 else len(lines)
        lines = lines[:max_sample]
        for line in tqdm(lines):
            caption = line.strip()
            captions.append(clip.tokenize(caption).to(device))
    
    return captions

def load_word_bank(filename: str, rt_type: type):
    word_bank = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = clip.tokenize(line.strip())[0]
            tokens = tokens[torch.nonzero(tokens)].squeeze().tolist()
            word_bank.append(tokens[1:-1][0])

    return rt_type(word_bank)

def contain(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return True
    return False    

def gender_attr(captions, word_bank_fis, size_parti=5):
    print("generate gender attribute")
    male_bank = load_word_bank(word_bank_fis['male'], set)
    female_bank = load_word_bank(word_bank_fis['female'], set)
    gender_attributes = []
    
    assert len(captions) % size_parti == 0, \
        "Length of captions should be divided by size_parti"
    num_parti = int(len(captions) / size_parti)
    for i in tqdm(range(num_parti)):
        gender_attr = []
        for j in range(size_parti):
            caption = captions[i * size_parti + j]
            if contain(caption, male_bank):
                gender_attr.append(Gender.Male)
            if contain(caption, female_bank):
                gender_attr.append(Gender.Female)
        if len(gender_attr) == 0:
            gender_attributes.append(Gender.Neutral)
        elif all([x == Gender.Male for x in gender_attr]):
            gender_attributes.append(Gender.Male)            
        elif all([x == Gender.Female for x in gender_attr]):
            gender_attributes.append(Gender.Female)
        else:
            gender_attributes.append(Gender.Neutral)
    gender_attributes = [x.value for x in gender_attributes]

    return gender_attributes

def neutralize_gender(captions, word_bank_fis):
    male_bank = load_word_bank(word_bank_fis['male'], list)
    female_bank = load_word_bank(word_bank_fis['female'], list)
    neutral_bank = load_word_bank(word_bank_fis['neutral'], list)
    num_repl_wd = min(
        [len(male_bank), len(female_bank), len(neutral_bank)]
    )
    
    replace_map = dict()
    for i in range(num_repl_wd):
        replace_map[male_bank[i]] = neutral_bank[i]
        replace_map[female_bank[i]] = neutral_bank[i]

    for i in tqdm(range(len(captions))):
        caption = captions[i]
        for wid in range(caption.shape[1]):
            if caption[0, wid].item() in replace_map:
                captions[i][0, wid] = replace_map[caption[0, wid].item()]

    return captions
