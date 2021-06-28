import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
# from modules import DeeplabV3
from modules import *


def make_model(opts, classes=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex
    
    '''
    # use a ResNet as backbone
    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    '''
    '''
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory
    '''
    '''
    # Initialize the head, which is usefull for the segmentation task.
    # In our opinion, this head is positioned after the body:
    # ( it takes as input the output channels of the body )
    head_channels = 256
    head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                     out_stride=opts.output_stride, pooling_size=opts.pooling)
    '''
    
    # we have to set the parameters:
    bisenet = BiSeNet(opts.num_classes, opts.backbone)  # use resente101 for the contextpath
    
    if classes is not None:
        model = IncrementalSegmentationModule(bisenet, opts.num_classes, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        # What is this class? We haven't it. Have we to implement it? We think no. We don't need it.
        model = SegmentationModule(bisenet, opts.num_classes, opts.num_classes, opts.fusion_mode)

    return model

# What does it do???
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, bisenet, numClasses, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.bisenet = bisenet
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        # create a list, where each element i has a conv layers having:
        # input dimension = head_channels
        # output dimension = number of classes for task c, c index = i
        # kernel = 1
        self.cls = nn.ModuleList(
            [nn.Conv2d(32, c, 1) for c in classes]  # cambiato da 'c' a 32
        )
        self.classes = classes
        self.numClasses = numClasses
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None
        
        self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=1)

    def _network(self, x):

        # take the input, put into bisenet
        x_b = self.bisenet(x)  # x_b contains probabilities
        # x_f0, 1, 2 contain features which represents the pixel codifications
        x_f0 = x_b[0]
        x_f1 = x_b[1]  # cx1
        x_f2 = x_b[2]  # cx2
        
        cx1_sup = self.supervision1(x_f1)
        cx2_sup = self.supervision2(x_f2)
        cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=x.size()[-2:], mode='bilinear')
        cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=x.size()[-2:], mode='bilinear')
        
        out_f0 = []
        out_f1 = []
        out_f2 = []
        
        # for each convolution in cls, add the result of the 
        # convolution applied on the output of FFM
        for mod in self.cls:
            out_f0.append(mod(x_f0))
            
        # concatenates the conv results
        # (out is a list of tensors)
        x_o_f0 = torch.cat(out_f0, dim=1)
        
        # for each convolution in cls, add the result of the 
        # convolution applied on the output of cx1
        for mod in self.cls:
            out_f1.append(mod(cx1_sup))
            
        # concatenates the conv results
        # (out is a list of tensors)
        x_o_f1 = torch.cat(out_f1, dim=1)
        
        # for each convolution in cls, add the result of the 
        # convolution applied on the output of cx2
        for mod in self.cls:
            out_f2.append(mod(cx2_sup))
            
        # concatenates the conv results
        # (out is a list of tensors)
        x_o_f2 = torch.cat(out_f2, dim=1)

        return x_o_f0, x_o_f1, x_o_f2, x_b

    def init_new_classifier(self, device):
        # take the last conv layer (conv layer of t-1)
        cls = self.cls[-1]
        
        # Hypotesis: Backgroun is the first class initialized, 
        #            due to that we take the elements in 0, in order to consider it.
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        # returns a tensor of the ( log ( |Ct| + 1 ) )
        # self.classes[-1] = |Ct|
        # we add 1, because the background class isn't considered in Ct
        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        # returns the features maps x_o_f0, x_o_f1, x_o_f2 and the probabilities x_b (the final output of BiSeNet)
        out = self._network(x)

        # take only x_o
        sem_logits_f0 = out[0]
        sem_logits_f1 = out[1]
        sem_logits_f2 = out[2]

        # up_sample x_o to the original dimension of the input
        sem_logits_f0 = functional.interpolate(sem_logits_f0, size=out_size, mode="bilinear", align_corners=False)
        sem_logits_f1 = functional.interpolate(sem_logits_f1, size=out_size, mode="bilinear", align_corners=False)
        sem_logits_f2 = functional.interpolate(sem_logits_f2, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits_f0, sem_logits_f1, sem_logits_f2, {"bisenet": out[3]}

        return sem_logits_f0, sem_logits_f1, sem_logits_f2, {}

    # fix batch normalization during training:
    def fix_bn(self):
        # iterates over the layers
        for m in self.modules():
            # if it is a batch normalization layer do:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
