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

import os
import random
import torch
import numpy as np

def is_cuda_available():
    return torch.cuda.is_available()

def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch1.backends.cudnn.deterministic = True
    # torch1.backends.cudnn.benchmark = True
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass
