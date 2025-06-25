from data_provider.data_factory import data_provider
from exp.exp_classification import Exp_Classification
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Stock_Classification(Exp_Classification):
    def __init__(self, args):
        super(Exp_Stock_Classification, self).__init__(args)

    def _build_model(self):
        self.args.pred_len = 0 # No prediction length for classification
        self.args.num_class = self.args.c_out
        
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
