import collections
import librosa, librosa.display
import soundfile as sf

import torch
from torch import Tensor
from torchvision import transforms
from torchaudio import transforms as audioTran
from torch.utils.data import Subset
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model

import os
import math
import yaml
import copy



class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        pre_trained_model_id = 'facebook/wav2vec2-xls-r-300m'
        self.processing = Wav2Vec2FeatureExtractor.from_pretrained(pre_trained_model_id)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(pre_trained_model_id)
        self.out_dim = 1024
    
        return

    def extract_feat(self, input_data): 
        emb = self.processing(input_data,sampling_rate=16000,padding=True,return_tensors="pt").input_values[0]
        emb = self.wav2vec2_model(emb.cuda()).last_hidden_state # [batch, 201, 1024] 

        return emb


class Classic_Attention(nn.Module):
    def __init__(self,embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.w = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        weights_view = self.w.unsqueeze(0).expand(inputs.size(0), len(self.w)).unsqueeze(2)
        attention_weights = F.tanh(inputs.bmm(weights_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized


class Model(nn.Module):
    def __init__(self, args,device, class_num = 3):
        super().__init__()
        self.device = device
        self.class_num = class_num

        self.first_dim = args['first_embedd_len']
        self.emb_dim = args['feature_embedd_len']


        self.ssl_model = SSLModel(self.device)
        self.dense1 = nn.Linear(self.ssl_model.out_dim, self.first_dim)
        self.bn1 = nn.BatchNorm1d(self.first_dim)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(self.first_dim, self.emb_dim)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.2)

        self.bn3 = nn.BatchNorm1d(self.emb_dim)
        self.relu3 = nn.LeakyReLU()
        
        ### Statistics attentive pooling
        self.attention = Classic_Attention(self.emb_dim)  ##frame-level attention
        self.dense3 = nn.Linear(self.emb_dim*2, self.emb_dim)

        self.classifier1 = nn.Linear(202, self.class_num)
        self.temporal_proj = nn.Linear(self.emb_dim, 1)
        
        self.att_weight = nn.Linear(self.class_num, self.class_num)  
        self.dense4 = nn.Linear(2, 1)

        # ########################### not used in this model
        # self.mel_transform = audioTran.MelSpectrogram(16000,n_fft=500)
        # self.to_DB = audioTran.AmplitudeToDB(top_db=80)
        
        # self.resNet = ResNet()
        
        # self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        # self.classifier3 = nn.Conv2d(256, class_num, 1, bias=False)
        
        # self.classifier4 = nn.Linear(256, 1)

    
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance
    
    def stat_attn_pool(self,inputs,attention_weights):   #adopt from https://blog.csdn.net/chengshrine/article/details/133798357
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
  
    def split_avg_pool(self, feature):   
        segment_sizes = [201,feature.shape[1]-201]
        segments = torch.split(feature, segment_sizes, dim=1)
        segments = [i.mean(dim=1, keepdim=True) for i in segments]
        segments = torch.cat(segments, dim=1)
        return segments


    
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        emb = self.ssl_model.extract_feat(x.squeeze(-1))

        x = self.dense1(emb)
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn1(x)
        x = x.view(org_size)  
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.dense2(x)
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn2(x)
        x = x.view(org_size)  
        x = self.relu2(x)
        x = self.drop2(x)

        
        attn_w = self.attention(x)
        x_utterance = self.stat_attn_pool(x,attn_w)
        x_utterance = self.dense3(x_utterance)
        feature_emb = torch.cat((x,x_utterance.unsqueeze(1)),1)   ### feature embeddings  (bs,202,128)

    
        ### learns the class activation values for the time feature   
        temporal_cams = self.classifier1(feature_emb.transpose(1,2))        # (batch,128,3)  
        temporal_out = self.temporal_proj(temporal_cams.transpose(1,2)).squeeze()
     
        ### maps the class activation values to an attention scalar value per time step
        temporal_cams = self.att_weight(temporal_cams)  #([batch, 128, 3])
   
        temporal_cams = F.softmax(temporal_cams, dim=-2)  #([batch, 128, 3]) 
        
        cams_feature = torch.matmul(feature_emb,temporal_cams)   # dotproduct att map back to temporal feature (batch,202,3)
        cams_feature = (temporal_out.unsqueeze(1)*cams_feature)   #(batch,202,3)
        cams_out = self.split_avg_pool(cams_feature) # (batch,2,3) 
        cams_out = self.dense4(cams_out.transpose(1,2)).squeeze()     # (batch,3)

        return temporal_out, cams_feature, cams_out
