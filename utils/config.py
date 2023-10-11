'''
    Project's main config.
'''

import os
from dataclasses import dataclass

@dataclass
class ConfigS:
    '''
        Project's main config.
    '''

    clip_model: str = 'openai/clip-vit-base-patch32'
    text_model: str = "stefan-it/german-gpt2-larger"
    seed: int = 42
    num_workers: int = 0
    # train_size: int = 0.84
    train_size: int = 0.9
    # val_size: int = 0.13
    val_size: int = 0.1
    epochs: int =10
    lr: int = 5e-6
    k: float = 0.33
    batch_size_exp: int = 2
    ep_len: int = 4
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 20
    dropout: float = 0.1
    weights_dir: str = os.path.join('weights', 'small_i3d_bs4_6layerencoder_eplen4_max_20_8_head_newloader')
    change: int = 65000
    slice_num: int = 5

@dataclass
class ConfigL:
    '''
        Project's main config.
    '''

    clip_model: str = 'openai/clip-vit-large-patch14'
    text_model: str = "stefan-it/german-gpt2-larger"
    seed: int = 100
    num_workers: int = 0
    train_size: int = 0.9
    val_size: int = 0.1
    epochs: int = 5
    lr: int = 5e-3
    k: float = 0.3
    batch_size_exp: int = 4
    ep_len: int = 4
    num_layers: int = 5        
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 20
    dropout: float = 0.08
    weights_dir: str = os.path.join('weights', 'large')
    slice_num: int = 5