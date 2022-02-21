import torch
import clip
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import pickle
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

train_num = 118287
indexes = [i for i in range(0, train_num)]
test_num = 5000

dataDir='..'
dataType='val2017'
save_folder='{}/feature/feature_{}.pkl'.format(dataDir,dataType)
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

save_json = []

def Neutralize(sent):
    sent = sent.lower()
    man_words =     ['man',   'men',   'boy',   'boys', 'gentleman',  'father', 'dad', 'son', 'male', 'husband', 'boyfriend', 'brother']
    woman_words =   ['woman', 'women', 'girl', 'girls',      'lady',  'mother', 'mom', 'daughter', 'female', 'wife', 'girlfriend', 'sister']
    replace_words = ['person', 'people', 'child', 'children','adult', 'parent', 'parent', 'child', '', 'spouse', 'partner', 'sibling', 'sibling']
    
    words_list = sent.split()
    new_sent = ''
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
        if i < len(words_list) -1:
            new_sent = new_sent + replace_w + ' '
        else:
            new_sent = new_sent + replace_w 
    return if_replace, new_sent    


for k in tqdm(coco.imgs.keys()):
    img = coco.imgs[k]
    #I = io.imread(img['coco_url'])

    img_name = ( str(img['id']) + '.jpg').zfill(16)
    img_path = '{}/images/{}/{}'.format(dataDir,dataType,img_name)
    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    anns = coco_caps.loadAnns(annIds)
    captions = []
    #print(len(anns))
    #assert len(anns) == 5
    
    for cap in anns:
        sent = cap['caption']
        if_replace, new_sent = Neutralize(sent)
        #if if_replace:
        #    print(sent)
        #    print(new_sent)
        captions.append(new_sent)

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize(captions).to(device)
    feature_vector = {}
    feature_vector['id'] = img['id']
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        feature_vector['image_feature'] = image_features.cpu().numpy()
        feature_vector['text_feature'] = text_features.cpu().numpy()
        #logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    save_json.append(feature_vector)
    #print("Label probs:", probs)


with open(save_folder, 'wb') as f:
    pickle.dump(save_json, f)