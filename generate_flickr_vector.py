import torch
import clip
from PIL import Image
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm
import operator
from flickr30k_entities_utils import get_sentence_data, get_annotations


def load_data(name):
    words_list = open(name, encoding="utf8")
    sentences = words_list.readlines()
    words_list.close()
    
    index_lits = []
    for s in sentences:
        index_lits.append(int(s))

    return index_lits


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
    

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


save_json = []
test_index = load_data("../flickr/flickr30k_entities/test.txt")
save_folder = "../feature/feature_flickr_test.pkl"


for i in tqdm(test_index):
    img_path = '../flickr/flickr30k-images/' + str(i) + '.jpg'
    #print(img_path)
    img = Image.open(img_path)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    #ann = get_annotations('../flickr/Annotations/' + str(i) + '.xml')
    sents = get_sentence_data('../flickr/Sentences/' + str(i) + '.txt')
    
    five_sentences = []

    for j in range(0, 5):
        sent = sents[j]['sentence']
        
        if_replace, new_sent = Neutralize(sent)
        #if if_replace:
        #    print(sent)
        #    print(new_sent)
        five_sentences.append(new_sent)

    text = clip.tokenize(five_sentences).to(device)
    feature_vector = {}
    feature_vector['id'] = i
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