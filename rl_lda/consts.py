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

# ========================================================

# ========================================================

LOG_FREQUENCY = 200

LOG_FORMAT = ("[%(asctime)s][%(filename)s - line %(lineno)s] "
              "- %(levelname)s - %(message)s")

# ========================================================

#

# ========================================================

CORL_VALID_PARAMS = {

    'encoder_name': ['transformer', 'lstm', 'mlp'],

    'decoder_name': ['lstm', 'mlp'],

    'reward_mode': ['episodic', 'dense'],

    'reward_score_type': ['BIC', 'BIC_different_var'],

    'reward_regression_type': ['LR','QR','GPR']
}

RL_VALID_PARAMS = {

    'encoder_type': ['TransformerEncoder'],

    'decoder_type': ['SingleLayerDecoder'],

    'decoder_activation': ['tanh', 'relu', 'none'],

    'score_type': ['BIC', 'BIC_different_var'],

    'reg_type': ['LR', 'QR', 'GPR']
}

GRANDAG_VALID_PARAMS = {

    'model_name': ['NonLinGaussANM', 'NonLinGauss'],

    'nonlinear': ['leaky-relu', 'sigmoid'],

    'optimizer': ['rmsprop', 'sgd'],

    'norm_prod': ['paths', 'none']
}

NOTEARS_VALID_PARAMS = {

    'loss_type': ['l2', 'logistic', 'poisson']
}

NONLINEAR_NOTEARS_VALID_PARAMS = {

    'model_type': ['mlp', 'sob']
}

MCSL_VALID_PARAMS = {

    'model_type': ['nn', 'qr']
}

DIRECT_LINGAM_VALID_PARAMS = {

    'measure': ['pwling' , 'kernel']
}

PC_VALID_PARAMS = {

    'variant': ['original', 'stable', 'parallel'],

    'ci_test': ['fisher', 'g2', 'chi2']
}

TTPM_VALID_PARAMS = {

    'penalty': ['BIC', 'AIC']
}

GNN_VALID_PARAMS = {

    'encoder_type': ['mlp', 'sem'],

    'decoder_type': ['mlp', 'sem'],

    'optimizer': ['adam', 'sgd']
}
