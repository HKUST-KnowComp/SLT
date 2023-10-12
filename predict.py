'''
    Script for single prediction on an image. It puts result in the folder.
'''

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch.utils.data import random_split, RandomSampler

from model import Net, InceptionI3d
from utils import ConfigS, ConfigL, download_weights
from transformers import GPT2Tokenizer, AutoTokenizer
from data import VideoReader,get_loader,cl_fn
from functools import partial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='model.pt',
    help='Checkpoint name'
)

parser.add_argument(
    '-S', 
    '--size',
    type=str,
    default='S',
    help='Model size [S, L]',
    choices=['S', 'L', 's', 'l']
)

parser.add_argument(
    '-I',
    '--img-path',
    type=str,
    default='',
    help='Path to the image'
)

parser.add_argument(
    '-R',
    '--res-path',
    type=str,
    default='./data/result/prediction',
    help='Path to the results folder'
)

parser.add_argument(
    '-T', 
    '--temperature',
    type=float,
    default=1.0,
    help='Temperature for sampling'
)

args = parser.parse_args()

# config = ConfigL() if args.size.upper() == 'L' else ConfigS()
config = ConfigS()
tokenizer_path = ''

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

path_to_visual_weight = ''
path_to_data = ''
path_to_align_weight = ''
path_to_save_plt = ''
if __name__ == '__main__':
    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)
    
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    
    tokenizer = AutoTokenizer.from_pretrained('stefan-it/german-gpt2-larger',pad_token = '<pad>',eos_token = '</s>',bos_token = '<s>',unk_token = '<unk>')
    visual_model = InceptionI3d(5383)

    state_dict = torch.load(path_to_visual_weight)['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove the 'module.' prefix
            new_state_dict[name] = v
    visual_model.load_state_dict(new_state_dict)
    visual_model = visual_model.to(device)
    dataset = VideoReader(path_to_data,tokenizer,device,visual_model,'openai/clip-vit-base-patch32')
    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [config.train_size, config.val_size, config.test_size])
    sampler = RandomSampler(val_dataset,replacement = False,num_samples = 1000)
    valid_loader = get_loader(
        val_dataset, 
        bs_exp= 0, #config.batch_size_exp if is_cuda else 2, 
        shuffle=False, 
        cl_fn = partial(cl_fn,model = visual_model,device=device),
        sampler = sampler,
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    cmap = plt.get_cmap('tab10')
    img_emb_lst = []
    img_mapped_lst = []
    tokenizer_lst = []
    model_lst = []
    tsne = TSNE(n_components=2,random_state = config.seed)
    transform = False
    root = path_to_align_weight
    tokenizer_dir = ''
    model_dir = ''
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(root,tokenizer_dir),pad_token = '<pad>',eos_token = '</s>',bos_token = '<s>')
    label_name = 'after'
    model = Net(
            clip_model=visual_model,
                text_model=config.text_model,
                tokenizer = tokenizer,
                ep_len=config.ep_len,
                num_layers=config.num_layers, 
                n_heads=config.n_heads, 
                forward_expansion=config.forward_expansion, 
                dropout=config.dropout, 
                max_len=config.max_len,
                device=device,
                slice_num=5
            )
    checkpoint = torch.load(os.path.join(root,model_dir))
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith('ie'):
            new_key = key.replace('ie', 'ie.model')
        else:
            new_key = key

        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    for idx, (img_emb, cap, att,is_video) in enumerate(valid_loader):
        if idx == 500:
            break
        img_emb = img_emb.to(device)
        with torch.no_grad():
            caption, _ ,img_mapped,img_embbed = model(img_emb, args.temperature)
            img_mapped = img_mapped.view(-1)
            img_embbed = img_embbed.view(-1)
            img_mapped_lst.append(img_mapped.to('cpu').numpy())
            img_emb_lst.append(img_embbed.to('cpu').numpy())
    tsne_proj = tsne.fit_transform(np.stack(img_mapped_lst))
    for i in range(100):
        if i ==0:
            plt.scatter(tsne_proj[i, 0], tsne_proj[i, 1],marker = '*',color = cmap(0),label = f"after_mapping",s =50)
        else:
            plt.scatter(tsne_proj[i, 0], tsne_proj[i, 1],marker = '*',color = cmap(0),s =50)
    tsne_proj = tsne.fit_transform(np.stack(img_emb_lst))
    for i in range(400):
        if i ==0:
            plt.scatter(tsne_proj[i, 0], tsne_proj[i, 1],marker = '+',color = cmap(1),label = f"before_mapping",s =50)
        else:
            plt.scatter(tsne_proj[i, 0], tsne_proj[i, 1],marker = '+',color = cmap(1),s =50)
    plt.title("T-SNE Visualization Of Video Embedding")
    plt.legend()
    plt.savefig(path_to_save_plt,bbox_inches='tight',dpi = 1000)