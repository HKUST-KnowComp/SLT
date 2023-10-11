from random import shuffle
import torch
import pandas as pd
import mediapipe as mp
import json
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer, AutoModel
import pysrt
import datetime
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoTokenizer
from tqdm import tqdm
import time
from math import ceil

EXPECTED_FRAME = 64

def unit_transfer(time):
    time = time.to_time()
    seconds = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond).total_seconds()
    return seconds

def create_srt(text,start_sec = 0,start_min = 0,start_h =0 ,end_sec = 1,end_min = 0,end_h = 0):# as the time is not important
    subs = pysrt.SubRipFile()

    # Add a new subtitle to the SRT file
    sub = pysrt.SubRipItem()
    sub.index = 1
    sub.start.seconds = start_sec
    sub.start.minutes = start_min
    sub.start.hours = start_h
    sub.end.seconds = end_sec
    sub.end.minutes = end_min
    sub.end.hours = end_h
    sub.text = text
    subs.append(sub)
    return subs



def parse_video(path,srt_path):
    subs = pysrt.open(srt_path)
    cap = cv2.VideoCapture(path)
    start_time_list = []
    start_frame_list = []
    end_time_list = []
    end_frame_list = []
    subtitle_lst = []
    for f in range(len(subs)):
        # print(subs[f].start)
        # print(unit_transfer(subs[f].start))
        start_time_list.append(unit_transfer(subs[f].start))
        end_time_list.append(unit_transfer(subs[f].end))   #get the start time and end time
        subtitle_lst.append(subs[f].text)
        # cap.set(cv2.CAP_PROP_POS_FRAMES,start*1000)
    for f in range(len(subs)):
        cap.set(cv2.CAP_PROP_POS_MSEC,start_time_list[f]*1000)
        start_frame_list.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        cap.set(cv2.CAP_PROP_POS_MSEC,end_time_list[f]*1000)
        end_frame_list.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    cap.release()
    frame_lst = []
    slice_lst = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = 0
    subtitle_id = 0
    while(cap.isOpened()):
        ret, frame =cap.read()
        if ret == False:
            break
        if count< start_frame_list[subtitle_id]:
            count+=1
            continue
        if count>end_frame_list[-1]:
            frame_lst.append(slice_lst)
            break
        if count>=start_frame_list[subtitle_id] and count<=end_frame_list[subtitle_id]:
            # img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # slice_lst.append(img)
            slice_lst.append(frame)
            count+=1
        else:
            count+=1
            subtitle_id+=1
            frame_lst.append(slice_lst)
            slice_lst = []
            continue
    
    # return zip([frame_lst[start_frame_list[0]:end_frame_list[0]] for i in range(len(subs))],subtitle_lst[0])
    return frame_lst,subtitle_lst,fps,width,height
    
def video_preprocess(path,path_to_video,path_to_subtitle,overwrite = False):
    if os.path.exists(path_to_video) and overwrite == False:
        print('path to video exist')
        return 
    if os.path.exists(path_to_subtitle) and overwrite == False:
        print('path to subtitle exist')
        return
    os.makedirs(path_to_video,exist_ok = True)
    os.makedirs(path_to_subtitle,exist_ok = True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #load the video path
    video_path_lst = []
    sub_path_lst = []
    for root,dir_lst,file_lst in os.walk(path):
        if len(dir_lst) == 0:
            if 'videos' in root:
                print('video root: ',root.split('.')[-1])
                for file in sorted(file_lst):
                    # if file == None:
                        # print('root: ',root)
                        # print('mp4 file: ',file)
                    video_path_lst.append(os.path.join(root,file))
            elif 'subtitle' in root:
                print('subtitle root: ',root.split('.')[-1])
                for file in sorted(file_lst):
                    sub_path_lst.append(os.path.join(root,file))
    assert len(video_path_lst) == len(sub_path_lst), 'the length of video lst and sub lst is not equal'
    #process the video
    for id in tqdm(range(len(video_path_lst)),desc = 'preprocess bar'):
        frame_lst,subtitle_lst,fps,width,height = parse_video(video_path_lst[id],sub_path_lst[id])
        slice_name = video_path_lst[id].split('/')[-1].split('.')
        sub_name = sub_path_lst[id].split('/')[-1].split('.')
        for i,frames in enumerate(frame_lst):
            out = cv2.VideoWriter(path_to_video+'/'+f'{i+1}.'+slice_name[-3]+'.'+slice_name[-2]+'.'+slice_name[-1], fourcc, fps, (int(width),int(height)))
            for frame in frames:
                out.write(frame)
            out.release()
            create_srt(subtitle_lst[i]).save(path_to_subtitle+'/'+f'{i+1}.'+sub_name[-2]+'.'+sub_name[-1])
    print('finished')

            
        
        
        
        
        

            
def encode_video(slices,model,device,resize = True,path = None,preprocessor = None):
    if resize == True:
        zero_pad = torch.zeros([224,224,3])
    elif resize == False:
        zero_pad = model(**preprocessor(images = torch.zeros([3,720,640]),return_tensors = 'pt').to(device)).pooler_output.to('cpu')
    count = 0
    current_slice = []
    for id,frame in enumerate(slices):
        if id%3 == 0:
            if resize == True:
                current_slice.append(cv2.resize(frame,(224,224)).tolist())
            elif resize == False:
                current_slice.append(model(**preprocessor(images = frame,return_tensors = 'pt').to(device)).pooler_output.to('cpu'))
    assert len(current_slice) != 0, print(path)
    current_slice = torch.tensor(current_slice)
    if len(current_slice) >= EXPECTED_FRAME:  #truncated and pad
        current_slice = current_slice[:EXPECTED_FRAME]
    else:
        pad_len = EXPECTED_FRAME - current_slice.shape[0]
        zeros = torch.stack([zero_pad for i in range(pad_len)])
        current_slice = torch.cat([current_slice,zeros])
    current_slice = current_slice.unsqueeze(0)
    current_slice = current_slice.permute(0,4,1,2,3)
    current_slice = current_slice.to(torch.float).to(device)
    return current_slice #[channel,frames,height,width]

class VideoReader(Dataset):
    def __init__(self,path,tokenizer,device,model,preprocessor,mode = 'train'):
        self.tokenizer = tokenizer
        self.device = device
        self.preprocessor = CLIPProcessor.from_pretrained(preprocessor)
        self.model = model
        self.mode = mode
        self.video_path_lst = []
        self.sub_path_lst = []
        self.video_lst = []
        self.sub_lst = []
        count = 0
        for root,dir_lst,file_lst in os.walk(path):
            if 'mp4' in root:
                for file in tqdm(sorted(file_lst),desc = 'reading mp4 file bar'):
                    if file.split('.')[-4] == '1' or file.split('.')[-4] == '0':
                        continue
                    self.video_path_lst.append(os.path.join(root,file))
            elif root.split('/')[-1] == 'subs':
                for file in tqdm(sorted(file_lst),desc = 'reading subs file bar'):
                    if file.split('.')[-3] == '1' or file.split('.')[-3] == '0':
                        continue
                    self.sub_path_lst.append(os.path.join(root,file))
                    
        assert len(self.video_path_lst)==len(self.sub_path_lst),'the length of subs and video path is not the same'
        
    def __len__(self):
        return len(self.video_path_lst)
    
    def read_video(self,path):
        frame_lst = []
        reader = cv2.VideoCapture(path)
        while(reader.isOpened()):
            ret,frame = reader.read()
            if ret  == False:
                break
            frame_lst.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        return frame_lst
    
    def __getitem__(self,idx):
        if self.mode == 'train':
            video_path = self.video_path_lst[idx]
            video = self.read_video(video_path)
            is_video = torch.Tensor([1]).to(self.device)
            sub_path = self.sub_path_lst[idx]
            subs = pysrt.open(sub_path)
            text = subs.text.replace('\n',' ')
            cap = self.tokenizer(text,padding = 'max_length',truncation = True,max_length = 16,return_tensors= 'pt')
            input_ids,attention_mask = cap['input_ids'].squeeze(dim  = 0),cap['attention_mask'].squeeze(dim = 0)
            if len(video) == 0:
                is_video = is_video = torch.Tensor([0]).to(self.device)
                vis_embedded = torch.zeros([1,3,64,224,224]).to(torch.float).to(self.device)
            else:
                vis_embedded = encode_video(video,self.model,self.device,True,video_path,self.preprocessor)
            
            return vis_embedded, input_ids, attention_mask,is_video
        else:
            return vis_embedded   
        
def cl_fn(batch, model,device):
    batch = list(zip(*batch))
    vis_embedded,input_ids,attention_mask,is_video = batch
    vis_embedded = torch.cat(vis_embedded).to(device)
    del batch

    video_embed = model(vis_embedded)['embds'].view(-1,1024)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    return video_embed, input_ids, attention_mask, is_video
        
def get_loader(dataset, cl_fn,bs_exp=5, shuffle=True, sampler = None):
    if sampler == None:
        return DataLoader(
            dataset,
            batch_size=2**bs_exp, 
            shuffle=shuffle,
            collate_fn=cl_fn
        )
    else:
        return DataLoader(
            dataset,
            batch_size=2**bs_exp, 
            shuffle=shuffle,
            collate_fn=cl_fn,
            sampler = sampler
        )
        