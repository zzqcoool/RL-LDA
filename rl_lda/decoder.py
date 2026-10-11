# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# Modified from the open-source RL implementation (Causal Discovery with Reinforcement Learning, ICLR 2020, arXiv:1906.04477; https://github.com/huawei-noah/trustworthyAI).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr

class SingleLayerDecoder(nn.Module):

    def __init__(self, batch_size, max_length, input_dimension, input_embed,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant, is_train, device=None):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = input_dimension
        self.input_embed = input_embed    # dimension of embedding space (actor)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant
        self.device = device
        self.is_training = is_train

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self._wl = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._wr = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._u = nn.Parameter(torch.Tensor(*(self.decoder_hidden_dim, 1)).to(self.device))
        self._l = nn.Parameter(torch.Tensor(1).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._wl)
        nn.init.xavier_uniform_(self._wr)
        nn.init.xavier_uniform_(self._u)

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = torch.randn([1]).numpy()[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        nn.init.constant_(self._l, bias_initial_value)

    def forward(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        W_l = self._wl
        W_r = self._wr
        U = self._u

        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, W_l)
        dot_r = torch.einsum('ijk, kl->ijl', encoder_output, W_r)

        tiled_l = torch.Tensor.repeat(torch.unsqueeze(dot_l, dim=2), (1, 1, self.max_length, 1))
        tiled_r = torch.Tensor.repeat(torch.unsqueeze(dot_r, dim=1), (1, self.max_length, 1, 1))

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = torch.einsum('ijkl, l->ijk', final_sum, U.view(self.decoder_hidden_dim))  # Readability

        self.logit_bias = self._l

        if self.use_bias:  # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        self.mask = 0
        self.samples = []
        self.mask_scores = []
        self.entropy = []

        for i in range(self.max_length):
            position = torch.ones([encoder_output.shape[0]],
                                  device=self.device) * i
            position = position.long()

            # Update mask
            self.mask = torch.zeros((encoder_output.shape[0], self.max_length),
                                    device=self.device).scatter_(1, position.view(encoder_output.shape[0], 1), 1)
            self.mask = self.mask.to(self.device)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad=True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy
