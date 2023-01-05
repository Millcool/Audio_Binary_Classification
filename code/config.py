import glob
import sys
import warnings

from scipy import stats
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
import copy
import time
import shutil, os
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import torchvision.models as models
from collections import defaultdict
warnings.filterwarnings("ignore")
import torchaudio
from IPython import display
import random
import wandb
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class CFG:
    JUST_PREDICT  = False
    Kaggle        = False
    DEBUG         = False
    FULL_DATA     = True
    wandb_on      = False
    seed          = 101
    MULTIMODEL    = False
    weights       = 'imagenet'
    backbone      = 'efficientnet-b1'
    model_name    = 'Docker'
    archive_name  = 'Audio'
    models        = []
    optimizers    = []
###################################################
    num_of_models = 1
    model_number  = 1
    train_bs      = 32
    valid_bs      = 32
    width         = 300 # image width
    mels          = 80  # height
    SAMPLE_RATE   = 16000
    NUM_SAMPLES   = 48000
    num_item_all  = 1000 if DEBUG else 45235     #611829  45235
    num_test      = 10 if DEBUG else 301      # 1000
    print_every   = 1  if DEBUG else 50      #500
    epochs        = 25  if DEBUG else 40        #35
    ###############################################
    crop_koef     = 1
    lr            = 0.002
    num_workers   = 4 if Kaggle else 0
    criterion     = nn.CrossEntropyLoss()
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    num_classes   = 2
    classes       = [0,1]
    activation    = None #'softmax'
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path     = " " if Kaggle else "E:/Audio/ASV/clips/"
    save_path     = '../working/result/' if Kaggle else "./result/"
    train_path    = "equal_audio/"
    csv_path      = 'equal_dataset'
    best_model_w  = '../input/russian-railways-2/best_epoch_ofu-efficientnet-b4_v2.bin' if Kaggle else f'./best_epoch_ofu-{backbone}_v2.bin'

