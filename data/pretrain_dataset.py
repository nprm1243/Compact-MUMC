import os
import argparse
import json
import random
import ruamel_yaml as yaml

from PIL import Image
from PIL import ImageFile

from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from cv2 import imread, imwrite
from data.utils import pre_question, pre_caption, pre_answer

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = pd.read_csv(ann_file)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        ann = self.ann.iloc[index, :]
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
        
        image_path = os.path.join(self.image_root, ann['name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, caption
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n