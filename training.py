'''
    Script that contains whole training process.
'''

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split

import wandb
from data import MiniFlickrDataset, get_loader,VideoReader,cl_fn
from model import Net, Trainer, InceptionI3d
from utils import ConfigS, ConfigL, LRWarmup
from transformers import AutoTokenizer
from functools import partial

parser = argparse.ArgumentParser()

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='',
    help='Checkpoint name'
)

parser.add_argument(
    '-S', 
    '--size',
    type=str,
    default='s',
    help='Model size [S, L]',
    choices=['S', 'L', 's', 'l']
)

args = parser.parse_args()

# config = ConfigL() if args.size.upper() else ConfigS()
config = ConfigS()


# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True


path_to_visual_weight = ''
path_to_data = ''
if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("stefan-it/german-gpt2-larger",pad_token = '<pad>',eos_token = '</s>',bos_token = '<s>',unk_token = '<unk>')
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

    train_loader = get_loader(
        train_dataset, 
        bs_exp=config.batch_size_exp if is_cuda else 2, 
        shuffle=True, 
        cl_fn = partial(cl_fn,model = visual_model,device=device),
    )

    valid_loader = get_loader(
        val_dataset, 
        bs_exp= 0, #config.batch_size_exp if is_cuda else 2, 
        shuffle=False, 
        cl_fn = partial(cl_fn,model = visual_model,device=device),
    )
    
    model = Net(
        clip_model=visual_model,
        text_model=config.text_model,
        tokenizer = tokenizer,
        ep_len=config.ep_len,
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        slice_num = config.slice_num,
        max_len=config.max_len,
        device=device
    )

    optimizer = optim.Adam(model.parameters(),lr = config.lr)#
    p_optimizer =  optim.Adam(model.parameters(), lr=config.lr)#pseudo

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    scheduler = optim.lr_scheduler.LambdaLR(p_optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()    

    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_dataset=test_dataset,
        test_path=os.path.join('data', 'raw', 'flickr30k_images'),
        ckp_path=ckp_path,
        device=device,
        config = config
    )

    for epoch in range(trainer.epoch, config.epochs):
        trainer.train_epoch()
        trainer.valid_epoch()

        metadata = trainer.get_training_data()
        trainer.save_ckp(os.path.join(config.weights_dir, f'epoch_{epoch + 1}.pt'))
        trainer.save_tokenizer((os.path.join(config.weights_dir, f'token_epoch_{epoch + 1}')))