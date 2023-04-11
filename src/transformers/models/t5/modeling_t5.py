# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """

import time

import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
import os
import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from ...activations import ACT2FN
from ...file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_t5 import T5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example::

            # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
            model = T5ForConditionalGeneration.from_pretrained('t5-3b')
            device_map = {0: [0, 1, 2],

                         1: [3, 4, 5, 6, 7, 8, 9],
                         2: [10, 11, 12, 13, 14, 15, 16],
                         3: [17, 18, 19, 20, 21, 22, 23]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example::

        # On a 4 GPU machine with t5-3b:
        model = T5ForConditionalGeneration.from_pretrained('t5-3b')
        device_map = {0: [0, 1, 2],

                     1: [3, 4, 5, 6, 7, 8, 9],
                     2: [10, 11, 12, 13, 14, 15, 16],
                     3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.is_decoder = config.is_decoder 

    def forward(self, hidden_states,  t=0):
        if self.training or not self.is_decoder:
            hidden_states = self.wi(hidden_states)
            hidden_states = nn.functional.relu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
        else:
            #input_shape=hidden_states.shape #[batch,seq_length,hidden_dim]
            #hidden_states= hidden_states.view(-1, hidden_states.size(-1)) 
            hidden_states = self.wi(hidden_states)
            hidden_states = nn.functional.relu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
            #expand the dim of output
            #hidden_states= hidden_states.view(-1, 1, hidden_states.size(-1)) 
            #hidden_states = hidden_states.expand(input_shape)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]
        self.is_decoder = config.is_decoder 

    def forward(self, hidden_states, t=0):
        if self.training or not self.is_decoder:
            hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
        else:
            #input_shape=hidden_states.shape 
            #hidden_states = hidden_states[:,t,:]
            hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
            #expand the dim of output
            #hidden_states= hidden_states.view(-1, 1, hidden_states.size(-1)) 
            #hidden_states = hidden_states.expand(input_shape)
        return hidden_states

#************************************#
#****Spiking Feed forward network****#
class T5DenseSpike(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.hidden = nn.Linear(config.d_ff, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = LinearSpike.apply
        self.is_decoder = config.is_decoder 
        self.leak = nn.Parameter(0.5*torch.ones(1))
        self.threshold = nn.Parameter(0.1*torch.ones(1))
        self.neg_act = LinearNegSpike.apply
        self.neg_threshold = nn.Parameter(-0.1*torch.ones(1))
        self.spiking_rate= 0.0
        self.local_recur = False
        self.local_window= 16
        self.spike_steps= 1 #4

    def forward(self, hidden_states, t=0):
        hidden_states = self.wi(hidden_states)
        #Spiking integrate and fire
        batch, seq_len, dim =hidden_states.shape
        if self.training==True:
            if self.local_recur==True:
                self.mem = torch.zeros_like(hidden_states).to(hidden_states.device)
                for i in range(self.local_window):
                    #TODO: put NN layer inside loop
                    #hidden_states = self.wi(hidden_states)
                    self.mem = self.leak * self.mem + hidden_states
                    mem_thr = self.mem / self.threshold - 1.0
                    rst = self.threshold * (mem_thr>0).float()
                    self.mem = self.mem - rst
                    #shift hidden states by 1 timestep
                    hidden_states = torch.cat((torch.zeros(batch,1,dim).to(hidden_states.device),hidden_states),dim=1)[:,:seq_len,:]
            else:
                self.mem = torch.zeros(batch,1,dim).to(hidden_states.device)
                #Note:keep a memory of full size
                #self.mem = torch.zeros_like(hidden_states).to(hidden_states.device)
                for i in range(seq_len):
                    self.mem = self.leak * self.mem + hidden_states[:,i,:].unsqueeze(1)
                    #input_til_now = torch.cat((hidden_states[:,:i+1,:],torch.zeros(batch,seq_len-1-i,dim).to(hidden_states.device)),dim=1)
                    
                    new_spike  = self.mem/ self.threshold - 1.0
                    rst = self.threshold * (new_spike>0).float()
                    self.mem = self.mem - rst
                    #negative spike
                    
                    neg_spike  = self.mem/ self.neg_threshold - 1.0
                    negrst = self.neg_threshold * (neg_spike<0).float()
                    self.mem = self.mem - negrst
                    
                    #only take the output of last timestep
                    if i==0:
                        mem_thr = new_spike
                        neg_mem = neg_spike
                    else: #last one
                        mem_thr = torch.cat((mem_thr, new_spike),dim=1)    
                        neg_mem = torch.cat((neg_mem, neg_spike),dim=1)      
                    
                    #Use a hidden gate for Vmem
                    #self.mem = self.hidden(self.mem)
                    
        else:
            if t==0:
                self.mem = torch.zeros_like(hidden_states).to(hidden_states.device)
        
            self.mem = self.leak * self.mem + hidden_states
            mem_thr = self.mem / self.threshold - 1.0
            rst = self.threshold * (mem_thr>0).float()
            self.mem = self.mem - rst
            #negative spike
            
            neg_mem  = self.mem/ self.neg_threshold - 1.0
            negrst = self.neg_threshold * (neg_mem<0).float()
            self.mem = self.mem - negrst
            
            #self.mem = self.hidden(self.mem)
            
        out = self.act(mem_thr)
        out = out + self.neg_act(neg_mem)
        #out = nn.functional.relu(mem_thr)
        #out = self.dropout(out)
        self.spiking_rate= torch.count_nonzero(out)/(out.shape[0]*out.shape[1]*out.shape[2])

        out = self.wo(out)
        return out

class T5LayerSpikingFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseSpike(config)
            logger.info("***** Using spike activation in FFN *****")
        elif  config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)

        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        self.is_decoder = config.is_decoder 
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states,t=0):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states,t)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif  config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)

        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        self.is_decoder = config.is_decoder 
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states,t=0):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states,t)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

#For spiking neural network
class AttentionSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """

    @staticmethod
    def forward(ctx, input):
        #input: bnqk
        #activate based on norm of each seq location
        batch_size = input.shape[0]
        n_dim = input.shape[1]
        #q_size = input.shape[2]
        k_size = input.shape[-1]
        i_norm = torch.mean(input, dim=1) #size= b1k or bk
        #***note: set a size
        select_size= min(k_size, max(4,int(k_size/2))) #min(64,k_size) #int(k_size/4)
        #i_norm = F.threshold(i_norm, threshold, 0)
        #print('mean=',torch.mean(i_norm))
        #TODO: select by threshold
        bool_select = i_norm > 0.5/k_size #mean =0.01
        count_select =  torch.sum(bool_select,dim=1)
        max_size = torch.max(count_select)
        select_size = min(max_size, select_size)
        #****TOPK*SELECTION, size of idx= b,(n*d), select_size *****# 
        values, select_idx = torch.topk(i_norm, select_size,dim=-1, sorted=False) #F.relu(i_norm- threshold)

        #TODO: add current?
        
        current_index = torch.full((batch_size,1), k_size-1)
        current_index.to(select_idx.device)
        select_idx = torch.cat((select_idx,current_index.to(select_idx.device)), dim=1)
        
        select_size= select_idx.size(1)
        #select_idx = i_norm>0 #torch.nonzero(i_norm)  #size= (s,3), s=number of nonzeros, 3=number of dim in index
        #print(i_norm.shape, select_idx.shape)
        ctx.save_for_backward(input,select_idx)
        out= torch.zeros(batch_size, n_dim, select_size).to(input.device)
        #out= torch.zeros(batch_size,n_heads, int(k_size/4), input.shape[3]).to(input.device)
        
        for i in range(batch_size):
            #for j in range(n_heads):
            out[i,:,:] = torch.index_select(input[i,:,:], dim=1, index=select_idx[i,:])        
        
        #print(out.shape,input.shape)
        return out, select_idx

    @staticmethod
    def backward(ctx, grad_output, idx):
        
        input,  select_idx   = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad = torch.zeros_like(input).to(input.device)
        #all_ones = torch.zeros_like(input).to(input.device)
        #grad[select_idx] = 1.0 
        batch_size = input.shape[0]
        #n_heads = input.shape[1]

        #grad[select_idx,:] = grad_input
        for i in range(batch_size):
            #for j in range(n_heads):
            for s in range(select_idx.shape[1]):
                grad[i,:, select_idx[i,s]] = grad_input[i,:,s]
        return grad, None

#For spiking neural network
class SequenceSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """

    @staticmethod
    def forward(ctx, input):
        #input: bnkd-> b(n*d)k
        #activate based on norm of each seq location
        batch_size = input.shape[0]
        n_dim = input.shape[1]
        k_size = input.shape[2]
        #TODO: use mean instead of norm because it's easy to compute and can be negative
        i_norm = torch.mean(input, dim=1)
        #i_norm = torch.norm(input, dim=1) #size= b1k or bk
        #***note: set a size
        '''
        select_size= min(k_size, max(8,int(k_size/2))) #if k<8, s=k; if k>16, s=k/2. if 8<k<16, k=8
        #i_norm = F.threshold(i_norm, threshold, 0)
        #TODO: select by threshold
        #i_norm = i_norm.reshape(batch_size,-1)
        #select_idx = i_norm > 1.0
        #print('mean of mean',torch.mean(i_norm))
        bool_select = i_norm > 0.0 #15.0 #mean of norm=11
        count_select =  torch.sum(bool_select,dim=1)
        max_size = torch.max(count_select)
        '''
        #select_size =  max(select_size, max_size)
        #TODO: set a constant for size
        select_size= min(k_size, 32)
        #select_size = min( max(max_size, int(k_size/4) ), select_size)
        #TODO: test full length
        #select_size = k_size
        #****TOPK*SELECTION, size of idx= b,(n*d), select_size *****# 
        values, select_idx = torch.topk(i_norm, select_size,dim=-1, sorted=False) #F.relu(i_norm- threshold)

        #TODO: add current?
        '''
        current_index = torch.full((batch_size,1), k_size-1)
        current_index.to(select_idx.device)
        select_idx = torch.cat((select_idx,current_index.to(select_idx.device)), dim=1)
        select_size= select_idx.size(1)
        '''
        #select_idx = i_norm>0 #torch.nonzero(i_norm)  #size= (s,3), s=number of nonzeros, 3=number of dim in index
        #print(i_norm.shape, select_idx.shape)
        ctx.save_for_backward(input,select_idx)
        out= torch.zeros(batch_size, n_dim, select_size).to(input.device)
        #out= torch.zeros(batch_size,n_heads, int(k_size/4), input.shape[3]).to(input.device)
        
        for i in range(batch_size):
            #for j in range(n_heads):
            out[i,:,:] = torch.index_select(input[i,:,:], dim=1, index=select_idx[i,:])        
        
        #print(out.shape,input.shape)
        return out, select_idx

    @staticmethod
    def backward(ctx, grad_output, idx):
        
        input,  select_idx   = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = torch.zeros_like(input).to(input.device)
        
        #all_ones = torch.ones_like(input).to(input.device)
        #grad[select_idx] = 1.0 
        batch_size = input.shape[0]
        #n_heads = input.shape[1]

        #grad[select_idx,:] = grad_input
        for i in range(batch_size):
            #for j in range(n_heads):
            for s in range(select_idx.shape[1]):
                grad[i,:, select_idx[i,s]] = grad_input[i,:,s]
        #equation 1: if out[input>0]=1
        #grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad, None

#For spiking neural network
class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0 #input[input > 0] #1.0 #TODO: binary spike or full-precision?
        #out[input> 1.0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        #equation 3:if out[input>1]=1, out[0<input<1]=input
        #grad = LinearSpike.gamma*F.threshold(1.0-torch.abs(2.0*input-1.0), 0, 0)

        #equation 2: if out[input>0]=input
        #grad = torch.zeros_like(input).cuda()
        #grad [input > 0]      = 1.0
        #grad       = LinearSpike.gamma*F.threshold(input, 0, 0)

        #equation 1: if out[input>0]=1
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class LinearNegSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input < 0] = -1.0 #input[input > 0] #1.0 #TODO: binary spike or full-precision?
        #out[input> 1.0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()

        #equation 1: if out[input<0]=1
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

#----linear transformer forward/backward-----#
class LinearTF(torch.autograd.Function):

    clip_gradient = 1.0 #TODO:avoid gradient exploding
    @staticmethod
    def forward(ctx, query, key, value):
        ctx.save_for_backward(query, key, value)
        #compute memory state Z=phi(K)*V
        batch_size= query.size(0)
        n_heads= query.size(1)
        dim= query.size(3)

        attn_output = torch.zeros_like(query).to(query.device)
        kv_state = torch.zeros(batch_size,n_heads,dim,dim).to(query.device)
        #set causal mask

        
        
        for qi in range(query.size(2)):    
            
            #S = sum_j=1^i (Kj*Vj)
            kv_state += torch.matmul(
                key[:,:,:,qi].unsqueeze(3), value[:,:,qi,:].unsqueeze(2) #bndk,bnkd->bndd
            ) 
            #COMPUTE similarity for attention
            attn_o = torch.matmul(query[:,:,qi,:].unsqueeze(2), kv_state) #bn1d, bndd -> bn1d
            attn_output[:,:,qi,:] = attn_o.squeeze(2)
        
        return attn_output

    
    @staticmethod
    def backward(ctx, grad_output):
        query, key, value  = ctx.saved_tensors
        q_grad = torch.zeros_like(query).to(query.device)
        k_grad = torch.zeros_like(key).to(query.device)
        v_grad = torch.zeros_like(value).to(query.device)
        batch_size= query.size(0)
        n_heads= query.size(1)
        dim= query.size(3)
        #init
        grad_input = grad_output.clone() #bnqd

        
        s = torch.zeros(batch_size,n_heads,dim,dim).to(query.device)
        for qi in range(query.size(2)):   
            #S = sum_j=1^i (Kj*Vj)
            s += LinearTF.clip_gradient * torch.matmul(
                key[:,:,:,qi].unsqueeze(3), value[:,:,qi,:].unsqueeze(2) #bndk,bnkd->bndd
            ) 
            #COMPUTE gradient for Query, dQ= GiS'= GKV #TODO: transpose??
            grad = torch.matmul(grad_input[:,:,qi,:].unsqueeze(2), s) #bn1d*bndd=bn1d
            q_grad[:,:,qi,:] = grad.squeeze(2)
        
        #do not forget to reinit s
        qg = torch.zeros(batch_size,n_heads,1,1).to(query.device) 
        for qi in range(query.size(2)-1,-1,-1): #reversed order   
            '''
            #S = sum_j=N-1^i (Qi*Gi)
            s += LinearTF.clip_gradient * torch.matmul(
                query[:,:,qi,:].unsqueeze(2).transpose(2,3), grad_input[:,:,qi,:].unsqueeze(2) #bnd1,bn1d->bndd
            ) 
            #TODO: transpose??
            #s= s.transpose(2,3)
            #COMPUTE gradient for Query
            #K grad ' = S*V= QGV = (kd)(dk)(kd)= kd
            grad2 = torch.matmul(s, value[:,:,qi,:].unsqueeze(2).transpose(2,3)) #bndd*bndk=bndk
            k_grad[:,:,:,qi] = grad2.squeeze(3)
            #V grad= (S'*K)'=K'*S = sum (QjKiGj)= (kd)(dk)(kd)= kd, k=1
            #grad = torch.matmul(key[:,:,:,qi].unsqueeze(3).transpose(2,3), s) #bnkd*bndd=bnkd
            #TODO: use einsum, qkg
            #kg += torch.einsum("bnkd, bndk, bnkd-> bnkd", query[:,:,qi,:].unsqueeze(2), key[:,:,:,qi].unsqueeze(3), )
            v_grad[:,:,qi,:] = grad.squeeze(2)

            '''
            #S = sum_j=N-1^i (Qi*Gi)
            qgi= torch.matmul(
                query[:,:,qi,:].unsqueeze(2), grad_input[:,:,qi,:].unsqueeze(2).transpose(2,3) #bn1d,bnd1->bn11
            ) 
            
            qg += LinearTF.clip_gradient * qgi
            
            #COMPUTE gradient for Query
            #K grad ' = S*V= QGV = (kd)(dk)(kd)= kd
            grad2 = torch.matmul(qg, value[:,:,qi,:].unsqueeze(2)) #bn11*bn1d=bn1d
            k_grad[:,:,:,qi] = grad2.squeeze(2)
            #V grad= SK = sum(QjGj) Ki= (kd)(kd)(dk)= kd, k=1
            grad = torch.matmul(qg, key[:,:,:,qi].unsqueeze(3).transpose(2,3)) #bnkk*bnkd=bnkd
            v_grad[:,:,qi,:] = grad.squeeze(2)
        return q_grad, k_grad, v_grad
    
#----------------------#

class MemNetwork(nn.Module):
    def __init__(self, dim):        
        super().__init__()
        self.inner_dim = dim
        self.input_gate = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.hidden_gate = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.out_gate = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.tanh= nn.Tanh()
    def forward(
        self,
        x,
        h
        ):
        x=self.input_gate(x)
        h=self.hidden_gate(h)
        out_h = self.tanh(x+h)
        out_x = self.out_gate(out_h)

        return out_x, out_h


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, use_mem=False, split_crossattn=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.use_mem = use_mem
        activate_sparse = False
        self.key_length = 32 #will set when running forward()
        self.real_key_length= 32
        self.sparse = use_mem and activate_sparse
        self.enable_recur= use_mem and False
        if use_mem==True: #spiking activation
            self.act_func 	= LinearSpike.apply #AttentionSpike.apply #SequenceSpike.apply #LinearSpike.apply #TODO: try to use relu
            #self.threshold  = Variable(torch.randn(1),requires_grad=True).cuda()
            self.recur_func= LinearTF.apply
            #self.threshold  = torch.nn.Parameter( 0.1*torch.ones(self.inner_dim),requires_grad=True)
            self.threshold  = torch.nn.Parameter( 0.1*torch.ones(1),requires_grad=True)
            self.leak  = torch.nn.Parameter( torch.ones(1),requires_grad=True)
            self.v_threshold  = torch.nn.Parameter( 0.1*torch.ones(1),requires_grad=True) #different thresholds for key and value
            self.local_connect  =torch.nn.Parameter(torch.randn(4),requires_grad=True)
            
            
            #TODO: set connection types. 0:recurrent, 1:cumulative ,2:direct
            self.connect_type= 0
            self.window_size= 16 #8 #5 #16
            self.enable_reset=False

            #self.mem_gate = MemNetwork(self.inner_dim)
            self.mem_gate = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
            #self.mem_gate = nn.Linear(self.key_value_proj_dim, self.key_value_proj_dim, bias=False)
            #self.mem_gate_v = nn.Linear(self.key_value_proj_dim, self.key_value_proj_dim, bias=False)
            #self.mem = nn.RNN(self.inner_dim, self.inner_dim, batch_first=True)
            #self.out_gate = nn.Linear(self.inner_dim, self.inner_dim, bias=False)

            #TODO: JULY 2022
            self.RNN = nn.RNN(self.inner_dim, self.inner_dim, batch_first=True)
            self.mem_func = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=1)
            self.act_sparse_func= SequenceSpike.apply#nn.Sequential(
                #nn.Linear(self.inner_dim, self.inner_dim, bias=False),
                #nn.ReLU())#b(nd)k->b(nd)s
            if self.enable_reset==True:
                logger.info(f"Resetting spiking mem= {self.threshold}")
            # Mesh TensorFlow initialization to avoid scaling before softmax
            #self.spike_linear = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_linear = nn.Linear(self.d_model, self.d_model, bias=False)
            self.v_linear = nn.Linear(self.d_model, self.d_model, bias=False)
        #self.spike_forget = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)


        #TODO:Set to true to split encoded features into segments for cross-attention
        self.split_crossattn = split_crossattn 
        self.gen_len = 128 #For cnn-dailymail,decode_size=128. 
        if self.split_crossattn==True:
            self.act_func 	= LinearSpike.apply 
            self.split_size= 64 #For cnn-dailymail, use 8 because encode_size=1024, decode_size=128. If split_size=1024, it's not split
            #self.RNN_key = nn.RNN(self.inner_dim, self.inner_dim, batch_first=True)
            #self.SNN_key = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
            #self.SNN_value = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
            #self.RNN_value = nn.RNN(self.inner_dim, self.inner_dim, batch_first=True)
            #self.mem_gate_k = nn.Linear(self.split_size, self.split_size, bias=False)
            #self.mem_gate_v = nn.Linear(self.split_size, self.split_size, bias=False)
            self.SNN_kv = nn.Linear(self.key_value_proj_dim, self.key_value_proj_dim, bias=False)
            #self.SNN_kv = nn.Linear(self.key_value_proj_dim*self.key_value_proj_dim, self.key_value_proj_dim*self.key_value_proj_dim, bias=False)
            #self.RNN_kv = nn.RNN(self.key_value_proj_dim*self.key_value_proj_dim, self.key_value_proj_dim*self.key_value_proj_dim, batch_first=True)
            self.recur_select= 0 #set to 0 for SNN, 1 for RNN, 2 for none      
            self.concat_mem = False
            self.enable_ksvs = True
            self.threshold  = torch.nn.Parameter( 0.1*torch.ones(1),requires_grad=True)
            #self.scale  = torch.nn.Parameter( 0.1*torch.ones(1),requires_grad=True)
            self.leak  = torch.nn.Parameter( 0.1*torch.ones(1),requires_grad=True)
            #self.v_leak  = torch.nn.Parameter( torch.ones(1),requires_grad=True)
            #self.v_threshold  = torch.nn.Parameter( 0.1*torch.ones(1),requires_grad=True)
            logger.info(f"Splitting encoded features in cross attention, size= {self.split_size}, recur_select={self.recur_select}")


        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        t=0,
        max_length=200,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length
        if t==0: #initialize spiking neurons
            if self.split_crossattn==True:
                self.old_idx = -1
                self.old_value_idx = -1

                if self.recur_select==0: #SNN
                    #self.key_mem = torch.zeros(batch_size, self.split_size, self.inner_dim).to(hidden_states.device)
                    self.snn_mem = torch.zeros(batch_size, self.n_heads, self.key_value_proj_dim, self.key_value_proj_dim).to(hidden_states.device)
                    #self.value_mem = torch.zeros(batch_size, self.split_size ,self.inner_dim).to(hidden_states.device)
                elif self.recur_select==1: #RNN
                    self.key_mem = torch.zeros(1, batch_size, self.inner_dim).to(hidden_states.device)
                    self.value_mem = torch.zeros(1, batch_size, self.inner_dim).to(hidden_states.device)
            if self.use_mem==True:
                self.recur_mem =torch.zeros_like(hidden_states).to(hidden_states.device)
                if self.training:
                    mem_length = seq_length
                    self.hidden_mem = None
                    self.key_mem = None 
                    self.key_z = torch.zeros(batch_size, self.n_heads,  self.key_value_proj_dim, 1).to(hidden_states.device)
                    self.value_mem = None
                else:
                    mem_length = 1
                    self.hidden_mem = torch.zeros(batch_size, 1,self.inner_dim).to(hidden_states.device)
                    self.value_mem = torch.zeros(batch_size, self.n_heads, 1,self.key_value_proj_dim).to(hidden_states.device)
                    self.key_mem = torch.zeros(batch_size, self.n_heads, 1, self.key_value_proj_dim).to(hidden_states.device)
                    self.key_z = torch.zeros(batch_size, self.n_heads, self.key_value_proj_dim, 1).to(hidden_states.device)

                # (batch_size, n_heads, key_length, dim_per_head)
                #TODO: enable spiking mem
                self.mem_h=None
                
                self.hidden_states_mem = torch.zeros(batch_size, mem_length,self.inner_dim).to(hidden_states.device)
                
                self.key_states_mem = torch.zeros(batch_size, mem_length,self.inner_dim).to(hidden_states.device)#torch.zeros(batch_size, self.n_heads, mem_length,self.key_value_proj_dim).to(hidden_states.device)
                self.value_states_mem = torch.zeros(batch_size, mem_length,self.inner_dim).to(hidden_states.device)#torch.zeros(batch_size, self.n_heads, mem_length,self.key_value_proj_dim).to(hidden_states.device)
                #***set recurrent states S=KV*****#
                #self.kv_state = torch.zeros(batch_size, self.n_heads,self.key_value_proj_dim,self.key_value_proj_dim).to(hidden_states.device)
                #self.qh = torch.zeros(1, batch_size, self.inner_dim).to(hidden_states.device)
            

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                #before shaping: (batch, seq, dim)
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            #use cache for generation, not used for training
            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    if self.use_mem== False: #do not use this if spiking neurons are used
                        hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    else:
                        hidden_states = hidden_states
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        def integrate_and_fire(input_states, hidden_states,t, mem_potential=None, mem_gate=None):
            # (batch_size, n_heads, key_length, dim_per_head)

            batch_size = hidden_states.shape[0]
            n_heads =hidden_states.shape[1]
            key_len =hidden_states.shape[2]
            dim =hidden_states.shape[3]
            self.threshold=self.threshold.to(hidden_states.device)
            self.local_connect=self.local_connect.to(hidden_states.device)
            if self.training==True:
                if self.connect_type==0:
                    #TODO: add a memory gate
                    hidden_states = input_states
                    '''
                    mem_gate_out= self.mem_gate(input_states)
                    mem_gate_out= torch.cat((torch.zeros(batch_size,n_heads,1,dim).to(mem_gate_out.device), mem_gate_out[:,:,:-1,:]),dim=2)
                    mem_gate_out = torch.cumsum(mem_gate_out,dim=2)
                    hidden_states = hidden_states + mem_gate_out
                    '''
                    #TODO:recurrent mem
                    cum_mem = torch.zeros_like(input_states)
                    mem_in = input_states
                    #hidden_states = input_states #torch.zeros_like(input_states).cuda()
                    for ti in range(1,self.window_size):
                        mem_gate_out = mem_gate(mem_in) #self.mem_gate(mem_gate_out)
                        mem_gate_out = torch.tanh(mem_gate_out)
                        mem_gate_out= torch.cat((torch.zeros(batch_size,n_heads,1,dim).to(mem_gate_out.device), mem_gate_out[:,:,:-1,:]),dim=2)
                        
                        #mem_gate_out= torch.cat((torch.zeros(batch_size,n_heads,ti,dim).to(mem_gate_out.device), mem_gate_out[:,:,:-ti,:]),dim=2)
                        cum_mem += mem_gate_out
                        mem_in = mem_gate_out
                    hidden_states = hidden_states +cum_mem
                    #recurrent mem + out gates
                    '''
                    hidden_mem = input_states
                    cum_mem = torch.zeros_like(input_states).cuda()
                    for ti in range(1,17):
                        mem_gate_out = self.out_gate(hidden_mem)
                        hidden_mem  = self.mem_gate(hidden_mem)
                        hidden_mem = torch.cat((torch.zeros(batch_size,n_heads,ti,dim).to(hidden_mem.device), hidden_mem[:,:,:-ti,:]),dim=2)
                        cum_mem += mem_gate_out

                    hidden_states = hidden_states + cum_mem
                    '''
                elif self.connect_type==1:
                    for c in range(4):
                        intermediate= input_states*self.local_connect[c]
                        if c>0:
                            intermediate= torch.cat((intermediate[:,:,c:,:], torch.zeros(batch_size,n_heads,c,dim).cuda()),dim=2)
                        hidden_states += intermediate
                elif self.connect_type==2:
                    hidden_states =  hidden_states + input_states
                    #hidden_states +=  input_states
                
                #TODO: reset
                if self.enable_reset==True:
                    rst 			= self.threshold* (input_states>self.threshold).float()
                    rst= torch.cat((torch.zeros(batch_size,n_heads,1,dim).to(rst.device), rst[:,:,:-1,:]),dim=2)
                    rst = torch.cumsum(rst,dim=2)
                    #shift reset to right by one timestep
                    hidden_states = hidden_states - rst

                #TODO: spiking activation
                #mem_thr 		= (hidden_states/self.threshold) - 1.0 
                #out 			= self.act_func(mem_thr)
                out = hidden_states
            else:
                #print(hidden_states[:,:,t,:].shape, input_states.shape, t)
                #current_state = torch.unsqueeze(hidden_states[:,:,t,:],dim=2)
                #hidden_states_update = current_state + input_states
                #hidden_states[:,:,t,:] = torch.squeeze(hidden_states_update,dim=2)
                #mem_thr 		= (hidden_states[:,:,:t+1,:]/self.threshold) - 1.0 
                if t==0:
                    hidden_states = input_states #+ self.out_gate(input_states)
                    mem_potential = mem_gate(input_states)
                    mem_potential = torch.tanh(mem_potential)
                else:
                    if self.connect_type==0:

                        #shift by time step
                        hidden_states = torch.cat((hidden_states, input_states), dim=2)
                        expand_dim = hidden_states.shape[2] - mem_potential.shape[2]
                        mem_out = torch.cat((torch.zeros(batch_size,n_heads,expand_dim,dim).to(input_states.device), mem_potential),dim=2)
                        hidden_states = hidden_states+ mem_out
                        #update membrane potential (memory)
                        mem_potential= torch.cat((mem_potential,input_states),dim=2)
                        if mem_potential.shape[2]>self.window_size:
                            mem_potential = mem_potential[:,:,-self.window_size:-1,:]
                        #all 16 mems pass through mem gate again
                        mem_potential = mem_gate(mem_potential)
                        mem_potential = torch.tanh(mem_potential)
                        '''
                        #concat past hidden states
                        mem_out = torch.sum(mem_potential, dim=2, keepdim=True)
                        mem_out = input_states+ mem_out
                        #update membrane potential (memory)
                        mem_potential= torch.cat((mem_potential,input_states),dim=2)
                        if mem_potential.shape[2]>=self.window_size:
                            mem_potential = mem_potential[:,:,-self.window_size:-1,:]
                        #all 16 mems pass through mem gate again
                        mem_potential = mem_gate(mem_potential)
                        mem_potential = torch.tanh(mem_potential)

                        hidden_states = torch.cat((hidden_states, mem_out), dim=2)
                        '''
                    elif self.connect_type==1:
                        for c in range(4):
                            intermediate= input_states*self.local_connect[c]
                            if t+c < hidden_states.shape[2]:
                                intermediate= torch.cat((intermediate[:,:,t+c:,:], torch.zeros(batch_size,n_heads,t+c,dim).cuda()),dim=2)
                                hidden_states += intermediate
                            else:
                                hidden_states= torch.cat((hidden_states, intermediate), dim=2)
                    elif self.connect_type==2:
                        hidden_states = torch.cat((hidden_states, input_states), dim=2)

                #TODO:spiking activation
                #mem_thr 		= (hidden_states/self.threshold) - 1.0 
                #out 			= self.act_func(mem_thr)
                
                #****tanh activation*****#
                out =hidden_states
                
                #TODO: reset
                if self.enable_reset==True:
                    rst 			= self.threshold* (input_states>self.threshold).float()
                    hidden_states = hidden_states - rst

            self.spiking_rate= torch.count_nonzero(out)/(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3])

            #rst 			= self.threshold[l]* (mem_thr>0).float()
            #self.mem[l] 	= self.leak*self.mem[l] + input_states - rst
            return out, hidden_states, mem_potential

        def recurrent_apply(input_states, hidden, t, mem_in=None, mem_h=None):
            # (batch_size, seq_length, dim)
            batch_size = hidden.shape[0]
            key_len =hidden.shape[1]
            dim =hidden.shape[2]
            self.threshold=self.threshold.to(hidden.device)
            if self.training==True:
                #TODO: add a memory gate
                hidden = input_states
                mem_gate_in = input_states
                cum_mem = torch.zeros_like(input_states).cuda()
                mem_h = torch.zeros_like(input_states).cuda()
                mem_potential =None
                spiking_rate=0.0
                for ti in range(1,self.window_size):
                    #mem_gate_out, mem_h = self.mem_gate(mem_gate_in, mem_h) 
                    mem_gate_out= self.mem_gate(mem_gate_in)
                    #mem_gate_out = torch.tanh(mem_gate_out)
                    #TODO: apply spiking activation
                    mem_thr 		= torch.div(mem_gate_out,self.threshold) - 1.0 
                    mem_gate_in			= self.act_func(mem_thr)
                    #mem_gate_in = mem_gate_out
                    spiking_rate += torch.count_nonzero(mem_gate_in)/(mem_gate_in.shape[0]*mem_gate_in.shape[1]*mem_gate_in.shape[2])
                    #print('spiking count=', spiking_rate)
                    mem_gate_in= torch.cat((torch.zeros(batch_size,1,dim).to(mem_gate_in.device), mem_gate_in[:,:-1,:]),dim=1)
                    
                    #mem_gate_in= torch.cat((torch.zeros(batch_size,ti,dim).to(mem_gate_in.device), mem_gate_in[:,:-ti,:]),dim=1)
                    cum_mem += mem_gate_in
                #TODO: try forgetting instead of accumulation
                #hidden = hidden - cum_mem
                hidden = hidden + cum_mem
                self.spiking_rate= spiking_rate/self.window_size
            else:
                if t==0:
                    
                    hidden = input_states #+ self.out_gate(input_states)
                    mem_out = self.mem_gate(input_states)
                    
                    #mem_h = torch.zeros_like(input_states)
                    #mem_out, mem_h = self.mem_gate(input_states, mem_h)
                    
                    mem_thr 		= (mem_out/self.threshold) - 1.0 
                    mem_potential			= self.act_func(mem_thr)
                else:
                    #current_state = torch.unsqueeze(hidden_states[:,:,-1,:],dim=2)
                    hidden = input_states
                    mem_sum =  torch.sum(mem_in,dim=1, keepdim=True)
                    #TODO: try forgetting instead of accumulation
                    #hidden = hidden - mem_sum
                    hidden = hidden + mem_sum
                    #update membrane potential (memory)
                    mem_in= torch.cat((mem_in,input_states),dim=1)
                    
                    if mem_in.shape[1]>=self.window_size:
                        mem_in = mem_in[:,-self.window_size:-1,:]
                    mem_out= self.mem_gate(mem_in)
                    '''
                    
                    tmp_zero = torch.zeros_like(input_states)
                    mem_h= torch.cat((mem_h,tmp_zero),dim=1)
                    if mem_in.shape[1]>=self.window_size:
                        mem_in = mem_in[:,-self.window_size:-1,:]
                        mem_h = mem_h[:,-self.window_size:-1,:]
                    #all 16 mems pass through mem gate again
                    mem_out, mem_h = self.mem_gate(mem_in, mem_h)
                    '''
                    #mem_potential = torch.tanh(mem_potential)
                    #TODO: apply spiking activation
                    mem_thr 		= torch.div(mem_out, self.threshold) - 1.0 
                    mem_potential			= self.act_func(mem_thr)
                    #mem_potential = mem_out
                    self.spiking_rate = torch.count_nonzero(mem_potential)/(mem_potential.shape[0]*mem_potential.shape[1]*mem_potential.shape[2])
                    #print('spiking rate=',self.spiking_rate, ',shape=', mem_potential.shape)
                    #print('threshold,mem_out mean,max,min', self.threshold, torch.mean(mem_out),torch.max(mem_out),torch.min(mem_out))
                    #print('mem_potential mean,max,min', torch.mean(mem_potential),torch.max(mem_potential),torch.min(mem_potential))
                    
            out = hidden 
            
            #TODO:spiking activation
            #mem_thr 		= (hidden_states/self.threshold) - 1.0 
            #out 			= self.act_func(mem_thr)
            #self.spiking_rate= torch.count_nonzero(out)/(out.shape[0]*out.shape[1]*out.shape[2])

            return out, hidden, mem_potential, mem_h

        def spike_apply(hidden_states, Vmem=None, threshold=1.0, linear_fn=None):
            # (batch_size, seq_length, dim)
            #Spiking integrate and fire
            batch, seq_len, dim =hidden_states.shape
            self.local_recur = False
            self.local_window = 16
            self.spike_steps = 1 #the number of spike steps per input

            hidden_states = linear_fn(hidden_states)
            if self.training==True:
                if self.local_recur==True:
                    Vmem = torch.zeros_like(hidden_states).to(hidden_states.device)
                    for i in range(self.local_window):
                        Vmem = self.leak * Vmem + hidden_states
                        mem_thr = Vmem / threshold - 1.0
                        rst = threshold * (mem_thr>0).float()
                        Vmem = Vmem - rst
                        #shift hidden states by 1 timestep
                        hidden_states = torch.cat((torch.zeros(batch,1,dim).to(hidden_states.device),hidden_states),dim=1)[:,:seq_len,:]
                else:
                    Vmem = torch.zeros(batch,1,dim).to(hidden_states.device)
                    for i in range(seq_len):
                        #for j in range(self.spike_steps):
                        Vmem = self.leak * Vmem + hidden_states[:,i,:].unsqueeze(1)
                        new_spike = Vmem / threshold - 1.0
                        rst = threshold * (new_spike>0).float()
                        Vmem = Vmem - rst
                        if i==0:
                            mem_thr = new_spike
                        else:
                            mem_thr = torch.cat((mem_thr, new_spike),dim=1)
            else:
                if t==0:
                    Vmem = torch.zeros_like(hidden_states).to(hidden_states.device)
                Vmem = self.leak * Vmem + hidden_states
                mem_thr = Vmem / threshold - 1.0
                rst = threshold * (mem_thr>0).float()
                Vmem = Vmem - rst

            #out = mem_thr #self.act_func(mem_thr)
            out = self.act_func(mem_thr)
            self.spiking_rate= torch.count_nonzero(out)/(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3])

            return out, Vmem

        def RAF_apply(input_states, Vmem, layer, threshold, leak, mem_layer=None):
            #Vmem = leak * (mem_layer(Vmem.transpose(-2,-1)).transpose(-2,-1)) + layer(input_states)
            #x=layer(input_states.view(batch_size, self.n_heads, -1))
            #x= x.reshape(batch_size, self.n_heads, self.key_value_proj_dim, -1)
            if layer==None:
                x=input_states
            else:
                x=layer(input_states)
            Vmem = leak * Vmem + x
            mem_thr = Vmem / threshold - 1.0
            rst = threshold * (mem_thr>0).float()
            Vmem = Vmem - rst

            out = nn.functional.relu(mem_thr)
            #out = self.act_func(mem_thr)
            '''
            #inner loop for snn
            x=layer(input_states)
            out= torch.zeros_like(input_states).to(input_states.device)
            for i in range(input_states.shape[1]):
                Vmem = leak * Vmem + x[:,i,:].unsqueeze(1)
                mem_thr = Vmem / threshold - 1.0
                rst = threshold * (mem_thr>0).float()
                Vmem = Vmem - rst
                out[:,i,:] = self.act_func(mem_thr).squeeze(1)
            '''
            self.spiking_rate= torch.count_nonzero(out)/(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3])

            return out, Vmem
            
        #add recurrent gate before proj layers
        
        #q_hidden_states = torch.clone(hidden_states)
        if self.use_mem: #disabled
            #batch, seq, dim
            #TODO: use SNN
            key_states, self.key_states_mem= spike_apply(hidden_states, self.key_states_mem, self.threshold, self.k_linear)
            value_states, self.value_states_mem= spike_apply(hidden_states, self.value_states_mem, self.v_threshold, self.v_linear)
            
            self.spiking_rate= torch.count_nonzero(key_states)/(key_states.shape[0]*key_states.shape[1]*key_states.shape[2])
            self.v_spiking_rate= torch.count_nonzero(value_states)/(key_states.shape[0]*key_states.shape[1]*key_states.shape[2])
            #hidden_states, self.hidden_states_mem= spike_apply(hidden_states, self.hidden_states_mem, self.threshold, self.spike_linear)
            
            #hidden_states, self.hidden_states_mem = spike_apply(hidden_states,self.hidden_states_mem, self.threshold)
            #hidden_states, self.hidden_states_mem, self.hidden_mem, self.mem_h = recurrent_apply(hidden_states, self.hidden_states_mem,t, self.hidden_mem, self.mem_h)
            #hidden_states, self.recur_mem = self.recurrent(hidden_states, self.recur_mem)
                # get key/value states
            key_states = project(
                key_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                value_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
            )
        else:
            # get key/value states
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
            )
        

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        

        #print qkv shape
        #if key_value_states is None:
        #    print('self attention size',query_states.shape, key_states.shape, value_states.shape)
        #else:
        #    print('cross attention size',query_states.shape, key_states.shape, value_states.shape)

        #self attention
        if key_value_states is None and self.use_mem==True:
            #TODO: enable IF neurons
            #if self.enable_recur==False:
            #key_states, self.key_states_mem, self.key_mem = integrate_and_fire(key_states, self.key_states_mem, t, self.key_mem, mem_gate=self.mem_gate)
            #value_states, self.value_states_mem, self.value_mem = integrate_and_fire(value_states, self.value_states_mem, t,self.value_mem, mem_gate=self.mem_gate_v)

            self.threshold.to(key_states.device)
            self.v_threshold.to(key_states.device)
            
        
        select_index =None 
        key_select = None
        real_key_length =key_length
    
        #-----------------------------#
        # Regular computation of attention
        #-----------------------------#
        # compute scores
        self.key_length = key_length
        self.real_key_length = real_key_length
        
        if self.training==True:
            self.gen_len =query_states.shape[2]
        #TODO: split encoder features into segments
        if self.split_crossattn==True:
            #pad with zero
            if key_states.shape[2] % self.split_size != 0:
                #print("****padding encoded keys****", key_states.shape)
                pad_size = self.split_size* math.ceil(key_states.shape[2]/self.split_size) - key_states.shape[2]
                key_states = nn.functional.pad(key_states, (0,0,0, pad_size), "constant",0)
                #print("****padded encoded keys****",pad_size, key_states.shape)
        
            #split into chunks and store in a list, bnkd-> q(bnmd), m=split_size
            key_states_split = torch.split(key_states, self.split_size, dim=2)
            #key_states_split = list(key_states_split)
            #key_states_ori = key_states_split.copy()

            query_states_split = torch.split(query_states, 1, dim=2) #check size match
            #multiply Q*K. ***Note: some keys are ignored if the length does not match
            if self.training==True:
                self.gen_len =len(query_states_split)
                #split_len= min(len(key_states_split), len(query_states_split))
                #print(len(key_states_split),type(key_states_split),type(key_states_split[0]))
                #SNN integrate and fire or RNN
                #key_mem_all = []
                '''
                for i in range(len(key_states_split)):

                    #bn1d
                    if self.recur_select==0: #SNN, unshape to bk(n*d)
                        #key_mem_all.append(self.key_mem)
                        key_states_split[i], self.key_mem = SNN_apply(unshape(key_states_split[i]), self.key_mem, self.SNN_key, self.threshold, self.leak)
                    elif self.recur_select==1:#RNN
                        key_states_split[i], self.key_mem = self.RNN_key(unshape(key_states_split[i]), self.key_mem)
                
                #reshape and concat
                if self.recur_select==0 or self.recur_select==1:
                    if self.concat_mem==True:
                        #key_states_split = [ torch.cat((shape(key_mem_all[i]), shape(key_states_split[i])),dim=2) for i in range(len(key_states_split))]
                        key_states_split = [ torch.cat((shape(key_states_split[i]), key_states_ori[i]),dim=2) for i in range(len(key_states_split))]
                    else:
                        key_states_split = [shape(key_states_split[i])+key_states_ori[i] for i in range(len(key_states_split))]
                else:
                    if self.concat_mem==True:
                        key_states_split = [ torch.cat((key_states_split[max(0,i-1)], key_states_ori[i]),dim=2) for i in range(len(key_states_split))]
                '''

                '''
                #TODO: use all segments for training
                key_states_all= torch.cat(key_states_split, dim=2)
                scores = torch.matmul(query_states, key_states_all.transpose(3, 2))
                '''
                scores = [ torch.matmul(
                    query_states_split[i], key_states_split[min(len(key_states_split)-1, int(i*len(key_states_split)/len(query_states_split)))].transpose(3, 2)
                ) for i in range(len(query_states_split))]
                
                scores = torch.stack(scores,dim=2)
                scores= torch.squeeze(scores, dim=-2)
                
            else: #inference
                split_idx= min( int(t*len(key_states_split)/self.gen_len), len(key_states_split)-1)
                
                #TODO:SNN integrate and fire or RNN
                #bn1d
                '''
                old_mem = self.key_mem.clone()
                
                if split_idx!=self.old_idx:
                    if self.recur_select==0: #SNN
                        key_states_split[split_idx], self.key_mem = SNN_apply(unshape(key_states_split[split_idx]), self.key_mem, self.SNN_key,self.threshold, self.leak)
                    elif self.recur_select==1:#RNN
                        key_states_split[split_idx], self.key_mem = self.RNN_key(unshape(key_states_split[split_idx]), self.key_mem)
                
                if self.recur_select==0 or self.recur_select==1:
                    if self.concat_mem==True:
                        key_states_split[split_idx] = torch.cat( (shape(old_mem),shape(key_states_split[split_idx])),dim=2)
                        #key_states_split[split_idx] = torch.cat( (shape(key_states_split[split_idx]), key_states_ori[split_idx]),dim=2)
                    else:
                        key_states_split[split_idx] = shape(key_states_split[split_idx])+ key_states_ori[split_idx]
                else:
                    if self.concat_mem==True:
                        key_states_split[split_idx] = torch.cat( (key_states_split[max(0,split_idx-1)], key_states_ori[split_idx]),dim=2)
                '''
                scores = torch.matmul(
                    query_states, key_states_split[split_idx].transpose(3, 2) )
                #self.old_idx = split_idx
            key_length = self.split_size
            self.key_length = key_length


        elif self.sparse==True: #TODO: convert to sparse matrix
            '''
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            ) 
            '''
            
            #activate based on norm of each seq location
            #current_key = key_states[:,:,-1,:].view(batch_size,self.n_heads,1,-1)
            #current_key = current_key.transpose(3,2)
            #TODO:compare with threshold
            key_states_th = key_states.clone() # / self.threshold -1.0
            
            key_states_th= unshape(key_states_th) #bndk->bk(nd)
            #key_states_th= key_states_th.reshape(batch_size,self.inner_dim, -1) #b(n*d)k

            key_states_th = key_states_th.transpose(1,2)

            key_select, select_index = self.act_sparse_func(key_states_th) #b(nd)k->b(nd)s
            key_select= key_select.reshape(batch_size,self.n_heads, self.key_value_proj_dim, -1) 
            #add the current state on the diag
            #key_states = torch.cat((key_select,current_key), dim=3)
            key_length = key_select.shape[3] 
            self.key_length = key_length
            #self.real_key_length = real_key_length
            scores = torch.matmul(
                query_states, key_select #bnqd,bnds->bnqs
            ) 
            
            #TODO:recurrent key
            #TODO: split for cross-attention 
            '''
            self.real_key_length = min(key_length,32)
            scores =torch.zeros((batch_size,self.n_heads, real_seq_length, self.real_key_length)).to(value_states.device)
            key_t= key_states_th
            for i in range(1):#real_seq_length
                #key_t = self.mem_func(key_t) #b(nd)k
                key_select, _ = self.act_sparse_func(key_t) #b(nd)k->b(nd)s
                #(bn1k,bnvd)->(bn1d)
                key_select= key_select.reshape(batch_size,self.n_heads, self.key_value_proj_dim, -1) 
                #print(query_states.shape, key_select.shape)
                #scores [:,:,i,:]= torch.matmul(query_states[:,:,i,:].unsqueeze(2), key_select).squeeze(2)
                scores = torch.matmul(query_states, key_select)
            key_t = key_t.reshape(batch_size,self.n_heads, self.key_value_proj_dim, -1) 
            key_states = key_t.transpose(2,3)
            '''
        else:
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, self.real_key_length), device=scores.device, dtype=scores.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, self.real_key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask.to(position_bias.device)  # (batch_size, n_heads, seq_length, key_length)
                
                #position_bias = position_bias[:,:,:,:key_length] + mask[:,:,:,:key_length].to(position_bias.device)  # (batch_size, n_heads, seq_length, key_length)
                #print('Check Mask', mask.shape,mask)
        
        if self.sparse==True: 
            '''
            scores += position_bias
            '''
            position_bias=  position_bias.reshape(batch_size, self.n_heads*hidden_states.size(1), self.real_key_length)
            select_bias = torch.zeros((batch_size, self.n_heads*hidden_states.size(1), key_length), device=scores.device, dtype=scores.dtype)
            for i in range(batch_size):
                select_bias[i,:,:] = torch.index_select(position_bias[i,:,:], dim=1, index=select_index[i,:]) 
            select_bias= select_bias.reshape(batch_size,self.n_heads, hidden_states.size(1), key_length)
            scores += select_bias
        #TODO: set special position bias for split encoder features into segments
        elif self.split_crossattn==True:
            #print(scores.shape, position_bias.shape)

            if position_bias.shape[-1]<scores.shape[-1]: #padding
                position_bias =nn.functional.pad(position_bias, (0,scores.shape[-1]-position_bias.shape[-1]), "constant",0)
            if self.training==True:
                #scores += position_bias
                
                for i in range(scores.shape[2]):
                    # im:(i+1)m
                    pidx = min(int(i*len(key_states_split)/scores.shape[2]), position_bias.shape[-1]//self.split_size-1)
                    pid = max(0,pidx-1)
                    if self.concat_mem==True:
                        pbias = torch.cat((position_bias[:,:,i, (pid)*self.split_size: (pid+1)*self.split_size], position_bias[:,:,i, (pidx)*self.split_size: (pidx+1)*self.split_size]),dim=-1)
                    else:
                        pbias =  position_bias[:,:,i, (pidx)*self.split_size: (pidx+1)*self.split_size]
                    #print(scores[:,:,i,:].shape,position_bias[:,:,i, pidx*self.split_size: (pidx+1)*self.split_size].shape)
                    if pbias.shape[-1]<scores.shape[-1]: #padding
                        pbias =nn.functional.pad(pbias, (0,scores.shape[-1]-pbias.shape[-1]), "constant",0)
                    scores[:,:,i,:] += pbias
                
            else:
                split_idx= min(int(t*len(key_states_split)/self.gen_len), len(key_states_split)-1) #int(t/self.split_size)
                sid= max(0, split_idx-1)
                if self.concat_mem==True:
                    pbias= torch.cat((position_bias[:,:,:, sid*self.split_size: (sid+1)*self.split_size],position_bias[:,:,:, (split_idx)*self.split_size: (split_idx+1)*self.split_size]),dim=-1)
                else:
                    pbias= position_bias[:,:,:, (split_idx)*self.split_size: (split_idx+1)*self.split_size]
                
                #print(scores.shape, position_bias.shape,split_idx)
                if pbias.shape[-1]<scores.shape[-1]: #padding
                    pbias =nn.functional.pad(pbias, (0,scores.shape[-1]-pbias.shape[-1]), "constant",0)
                scores += pbias
        else:
            #key_length = scores.shape[3]
            #position_bias= position_bias[:,:,:,-key_length:]
            #print(scores.shape, position_bias.shape)
            scores += position_bias

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask


        #TODO: split encoder features into segments
        if self.split_crossattn==True:

            #pad with zero
            if value_states.shape[2] % self.split_size != 0:
                pad_size = self.split_size* math.ceil(value_states.shape[2]/self.split_size) - value_states.shape[2]
                value_states = nn.functional.pad(value_states, (0,0,0, pad_size), "constant",0)
            
            #split into chunks and store in a list
            value_states_split = torch.split(value_states, self.split_size, dim=2)
            #value_states_split = list(value_states_split)
            #value_states_ori = value_states_split.copy()
            #check size match
            assert len(value_states_split)== len(key_states_split)

            if self.training==True:
                attn_weights_split = torch.split(attn_weights, 1, dim=2)#bnqk->q(bnqk)
                #split_len= min(len(value_states_split), len(attn_weights_split))
                #TODO: multiply Ks*Vs, then approximate the missing part
                if self.enable_ksvs==True:
                    attn_output = []
                    attn_complement= []
                    self.kv_mem = torch.matmul(key_states.transpose(2,3), value_states)
                    #TODO: precompute linear(ksvs), range=num of segments
                    ksvs = [self.SNN_kv(torch.matmul(key_states_split[j].transpose(2,3), value_states_split[j])) for j in range(len(value_states_split))] #bnds * bnsd= m[bndd]
                    
                    for i in range(len(attn_weights_split)): #range =  q
                        idx= min(len(value_states_split)-1, int(i*len(value_states_split)/len(attn_weights_split)))
                        
                        #if self.old_idx != idx: #skip if segment idx not changed
                        #    ksvs = torch.matmul(key_states_split[idx].transpose(2,3), value_states_split[idx]) #bnds * bnsd= bndd
                        
                        #TODO: use SNN here
                        if self.recur_select==0:
                            #kmvm, self.kv_mem = SNN_apply(ksvs, self.kv_mem, self.SNN_kv, self.threshold, self.leak)
                            #TODO: use precomputed ksvs
                            kmvm = self.kv_mem - ksvs[idx]
                            #RAF
                            kmvm, self.snn_mem = RAF_apply(kmvm, self.snn_mem,None, self.threshold, self.leak)
                            
                            #ksvs, self.snn_mem = SNN_apply(ksvs, self.snn_mem, self.SNN_kv, self.threshold, self.leak)
                            #kmvm = torch.where(kv_mask>0, kmvm, kv_mask) #kmvm=torch.mul(ksvs, kmvm)
                            attn_complement = torch.matmul(query_states_split[i], kmvm) #bn1d * bndd = bn1d
                            #TODO: Test if use kv instead of kmvm
                            #attn_complement = torch.matmul(query_states_split[i], self.kv_mem) #bn1d * bndd = bn1d

                            #scale it by dividing norm of K
                            attn_complement = torch.div(attn_complement, torch.norm(key_states, dim=2).unsqueeze(2))
                            #TODO TEST: divide by q*norm(k)^T; bn1d/ (bn1d*bnd1)
                            #norm_factor = torch.matmul(query_states_split[i], torch.norm(key_states, dim=2).unsqueeze(2).transpose(2,3) ) 
                            #attn_complement = torch.div(attn_complement, norm_factor)
                            #attn_complement, self.snn_mem = SNN_apply(attn_complement, self.snn_mem, self.SNN_kv, self.threshold, self.leak)
                        elif self.recur_select==1:
                            kmvm = self.kv_mem - self.ksvs
                            attn_complement = torch.matmul(query_states_split[i], kmvm) #1d * dd = 1d
                            attn_complement, self.rnn_mem = self.RNN_kv(attn_complement, self.rnn_mem)
                        else:
                            kmvm = self.kv_mem - self.ksvs
                            attn_complement = torch.matmul(query_states_split[i], kmvm)#scale it by dividing norm of K
                            attn_complement = torch.div(attn_complement, torch.norm(key_states, dim=2).unsqueeze(2))
                            
                        #TODO: test: only using complement, no KsVs
                        attn_output.append( torch.matmul(attn_weights_split[i], value_states_split[idx]) + attn_complement )
                        #attn_output.append(  attn_complement )
                        self.old_idx = idx
                else:
                    attn_output = [ torch.matmul(
                        attn_weights_split[i], value_states_split[min(len(value_states_split)-1, int(i*len(value_states_split)/len(attn_weights_split)))]
                    ) for i in range(len(attn_weights_split))]

                '''
                #SNN integrate and fire or RNN
                #value_mem_all = []
                for i in range(len(value_states_split)):
                    #bn1d
                    #value_mem_all.append(self.value_mem)
                    if self.recur_select==0: #SNN
                        value_states_split[i], self.value_mem = SNN_apply(unshape(value_states_split[i]), self.value_mem, self.SNN_value, self.v_threshold, self.v_leak)
                    elif self.recur_select==1:#RNN
                        value_states_split[i], self.value_mem = self.RNN_value(unshape(value_states_split[i]), self.value_mem)
                
                #reshape and concat the memory with current input
                if self.recur_select==0 or self.recur_select==1:
                    if self.concat_mem==True:
                       # value_states_split = [ torch.cat((shape(value_mem_all[i]),shape(value_states_split[i])),dim=2) for i in range(len(value_states_split))]
                        value_states_split = [ torch.cat((shape(value_states_split[i]), value_states_ori[i]),dim=2) for i in range(len(value_states_split))]
                    else:
                        value_states_split = [ shape(value_states_split[i])+ value_states_ori[i] for i in range(len(value_states_split))]
                else:
                    if self.concat_mem==True:
                        value_states_split = [ torch.cat((value_states_split[max(0,i-1)], value_states_ori[i]),dim=2) for i in range(len(value_states_split))]

                #TODO: use all segments for training
                #value_states_all= torch.cat(value_states_split, dim=2)
                #attn_output = torch.matmul(attn_weights, value_states_all)
                '''
                
                attn_output = torch.stack(attn_output, dim=2)
                attn_output = torch.squeeze(attn_output, dim=-2)
            
            else:
                split_idx= min(int(t*len(value_states_split)/self.gen_len), len(key_states_split)-1)
                if t==0:
                    self.kv_mem = torch.matmul(key_states.transpose(2,3), value_states)
                '''
                old_value_mem = self.value_mem.clone()
                #TODO:SNN integrate and fire or RNN
                if self.old_value_idx!=split_idx:
                    if self.recur_select==0: #SNN
                        value_states_split[split_idx], self.value_mem = SNN_apply(unshape(value_states_split[split_idx]), self.value_mem, self.SNN_value, self.v_threshold, self.v_leak)
                    elif self.recur_select==1:#RNN
                        value_states_split[split_idx], self.value_mem = self.RNN_value(unshape(value_states_split[split_idx]), self.value_mem)
                if self.recur_select==0 or self.recur_select==1:
                    if self.concat_mem==True:
                        #print(old_value_mem.shape,value_states_split[split_idx].shape)
                        value_states_split[split_idx]= torch.cat((shape(old_value_mem), shape(value_states_split[split_idx])),dim=2)
                        #value_states_split[split_idx]= torch.cat((shape(value_states_split[split_idx]),value_states_ori[split_idx]),dim=2)
                    else:
                        value_states_split[split_idx]= shape(value_states_split[split_idx])+value_states_ori[split_idx]
                else:
                    if self.concat_mem==True:
                        value_states_split[split_idx]= torch.cat((value_states_split[max(0,split_idx-1)],value_states_ori[split_idx]),dim=2)
                '''
                if self.enable_ksvs==True:
                    #Approximate by a complementary matrix multiplicaion
                    if self.old_idx != split_idx: #skip if segment idx not changed
                        self.ksvs = torch.matmul(key_states_split[split_idx].transpose(2,3), value_states_split[split_idx]) #bnds * bnsd= bndd
                    #TODO: use SNN here
                    if self.recur_select==0:
                        #kmvm, self.kv_mem = SNN_apply(ksvs, self.kv_mem, self.SNN_kv, self.threshold, self.leak)
                        
                        kmvm = self.kv_mem - self.SNN_kv(self.ksvs)
                        kmvm, self.snn_mem = RAF_apply(kmvm, self.snn_mem, None, self.threshold, self.leak)
                        
                        
                        attn_complement = torch.matmul(query_states, kmvm) #1d * dd = 1d
                        #TODO: Test if use kv instead of kmvm
                        #attn_complement = torch.matmul(query_states, self.kv_mem) #1d * dd = 1d

                        #scale down 
                        attn_complement = torch.div(attn_complement, torch.norm(key_states, dim=2).unsqueeze(2))
                        #TODO TEST: divide by q*norm(k)^T; bn1d/ (bn1d*bnd1)
                        #norm_factor = torch.matmul(query_states, torch.norm(key_states, dim=2).unsqueeze(2).transpose(2,3) ) 
                        #attn_complement = torch.div(attn_complement, norm_factor)
                        #attn_complement, self.snn_mem = SNN_apply(attn_complement, self.snn_mem, self.SNN_kv, self.threshold, self.leak)
                    elif self.recur_select==1:
                        kmvm, self.kv_mem = self.RNN_kv(self.ksvs.view(batch_size, self.n_heads, -1), self.kv_mem.view(batch_size, self.n_heads, -1))
                        self.kv_mem = self.kv_mem.reshape(batch_size, self.n_heads, self.key_value_proj_dim,self.key_value_proj_dim)
                        kmvm = kmvm.reshape(batch_size, self.n_heads, self.key_value_proj_dim,self.key_value_proj_dim)
                    else:
                        kmvm = self.kv_mem - self.ksvs
                        attn_complement = torch.matmul(query_states, kmvm)
                        attn_complement = torch.div(attn_complement, torch.norm(key_states, dim=2).unsqueeze(2))

                    attn_output = torch.matmul( attn_weights, value_states_split[split_idx] ) +attn_complement
                    #TODO: test: only using complement, no ksvs
                    #attn_output =  attn_complement
                else:
                    attn_output = torch.matmul(
                        attn_weights, value_states_split[split_idx] )
                self.old_idx = split_idx
        elif self.sparse==True: #TODO: convert to sparse matrix
            #TODO: reduce length of attention weights from bnqk to bnqs
            '''
            attn_weights_th = attn_weights.reshape(batch_size,self.n_heads*hidden_states.size(1), key_length)
            attn_weights_th = attn_weights_th / self.threshold
            attn_select, select_index = self.act_func(attn_weights_th)
            attn_select = attn_select.reshape(batch_size,self.n_heads,hidden_states.size(1), -1)
            #print(attn_select.shape, attn_weights.shape)
            self.key_length = attn_select.size(3)
            self.real_key_length = real_key_length
            '''

            #activate based on key states
            #current_value = value_states[:,:,-1,:].view(batch_size,self.n_heads,1,-1)
            #current_value = current_value.transpose(3,2)
            
            value_states_th = value_states.clone() #/ self.v_threshold
            value_states_th= unshape(value_states_th)#bnvd->bv(n*d)
            #value_states_th= value_states_th.reshape(batch_size,self.inner_dim, -1) #b(n*d)v
            #print(value_states_th.shape)
            value_states_th = value_states_th.transpose(1,2)
            '''
            #print(self.real_key_length, value_states_th.shape, key_states_th.shape, select_index.shape)
            #TODO: set length
            #value_select=torch.zeros((batch_size,self.inner_dim, attn_select.size(3))).to(value_states.device)
            value_select=torch.zeros((batch_size,self.inner_dim, key_length)).to(value_states.device)

            for i in range(batch_size):
                #for j in range(self.n_heads):
                value_select[i,:,:] = torch.index_select(value_states_th[i,:,:], dim=1, index=select_index[i,:])  
            value_select= value_select.reshape(batch_size,self.n_heads, self.key_value_proj_dim , -1)
            #add the current state on the diag
            #value_states = torch.cat((value_select,current_value), dim=3)
            attn_output = torch.matmul(attn_weights, value_select.transpose(3, 2)) 
            '''
            #TODO:multiply with selected attention weights
            attn_output=torch.zeros((batch_size,self.n_heads, real_seq_length, self.key_value_proj_dim)).to(value_states.device)
            value_t= value_states_th
            #recurrect spiking
            for i in range(1):
                #value_t = self.mem_func(value_t)
                value_select, _= self.act_sparse_func(value_t)
                #(bn1k,bnvd)->(bn1d)
                value_select= value_select.reshape(batch_size,self.n_heads, self.key_value_proj_dim , -1)
                #(bnqs,bnsd)->(bnqd)
                #print(attn_weights.shape, value_select.shape)
                #attn_output[:,:,i,:]= torch.matmul(attn_weights[:,:,i,:].unsqueeze(2), value_select.transpose(3, 2)).squeeze(2)
 
                attn_output= torch.matmul(attn_weights, value_select.transpose(3, 2))
            #attn_output = torch.matmul(attn_select, value_select.transpose(3, 2)) 
            #value_t = value_t.reshape(batch_size,self.n_heads, self.key_value_proj_dim, -1) 
            #value_states = value_t.transpose(2,3)
            
            #TODO: add a residual of Q
            '''
            query_res, self.qh = self.RNN(unshape(query_states), self.qh)
            query_res = shape(query_res)
            attn_output = attn_output + query_res
            '''
        else:
            attn_output = torch.matmul(attn_weights, value_states) #(bnqk,bnvd)->(bnqd)



        attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        #print('attn_output size', attn_output.shape)
        #print('q,k,v size', query_states.shape,key_states.shape, value_states.shape)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, use_mem=False, split_crossattn=False): 
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias, use_mem=use_mem, split_crossattn=split_crossattn)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        t=0,
        max_length=200,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            t=t,
            max_length=max_length,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        #TODO: set split_crossattn to true
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False, split_crossattn=True)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        t=0,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            t=t,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.use_mem = False #TODO: use spiking networks or not
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias,use_mem=(self.is_decoder and self.use_mem)))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        t=0,#for decoder
        max_length=200,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        #start_time = time.time()
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            t=t,
            max_length=max_length,
        )

        #end_time = time.time()
        #logger.info(f"self attention time={end_time-start_time}")
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            #start_time = time.time()
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                t=t,
            )
            #end_time = time.time()
            #logger.info(f"cross_attention time={end_time-start_time}")
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, t=t)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5SpikingBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.use_mem = True #TODO: use spiking networks or not, if it's true, self-attention will not concatenate prev values
        self.use_selfattn = False
        if self.use_selfattn:
            self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias,use_mem=(self.is_decoder and self.use_mem)))

        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        #self.layer.append(T5LayerFF(config))
        self.layer.append(T5LayerSpikingFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        t=0,#for decoder
        max_length=200,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
        
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        

        if self.use_selfattn:
            self_attention_outputs = self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=self_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                t=t,
                max_length=max_length,
            )

            #end_time = time.time()
            #logger.info(f"self attention time={end_time-start_time}")
            #output of self attention: (attn_output,) + (present_key_value_state,) + (position_bias,)

            hidden_states, present_key_value_state = self_attention_outputs[:2]
            attention_outputs = self_attention_outputs[2:] #bias
        else:
            
            present_key_value_state = (None,None)
            attention_outputs = (None,)
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            cross_id= int(self.use_selfattn==True)
            
            if present_key_value_state is not None and self.use_selfattn:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = t+1

            #start_time = time.time()
            cross_attention_outputs = self.layer[cross_id](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                t=t,
            )
            #end_time = time.time()
            #logger.info(f"cross_attention time={end_time-start_time}")
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, t=t)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        if self.is_decoder:
            self.block = nn.ModuleList(
                []
            )
            for i in range(int(config.num_layers)):
                #self.block.append(T5SpikingBlock(config, has_relative_attention_bias=bool(i == 0)))
                self.block.append(T5Block(config, has_relative_attention_bias=bool(i == 0)))
        else:
            self.block = nn.ModuleList(
                [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(int(config.num_layers))]
            )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        t=0,#time step for generation
        max_length=200,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[len(self.block)-1][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    t=t,
                    max_length=max_length,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            #TODO: use separate position biases for spiking/non-spiking layers
            #if i== int(self.config.num_layers/2):
            #    position_bias = None
            #else:
            position_bias = layer_outputs[2]
            
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a text-to-text
    denoising generative setting.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            `What are input IDs? <../glossary.html#input-ids>`__

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            T5 uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at `T5 Training
            <./t5.html#training>`__.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`:
            `attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a
            sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.

        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

            >>> # forward pass
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            t=t,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        t=0,
        max_length=200,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            t=t,#Yinghan: add current time step as input to decoder
            max_length=max_length,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        max_length=200,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            #print('Prepare inputs',input_ids.shape)
            input_ids = input_ids[:, -1:]
            #print('Cut inputs',input_ids.shape)

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "max_length": max_length,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
