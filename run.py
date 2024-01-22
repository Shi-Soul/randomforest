# Import Packages

import logging
import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model import *

logging.basicConfig(filename='output.log',level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logging.info(f'-----------------------------\nStart of running with args: \"{" ".join(sys.argv[1:])}\"')

## Set Config

config = parse(" ".join(sys.argv[1:]))
info = f"Config: {vars(config)}"
if config.verbose:
    print(info)
logging.info(info)

model_config = get_model_config(config)

# Load Dataset

dataset,dataset_shape = get_dataset(config)
train_dataset,val_dataset,test_dataset = split_dataset(dataset,config.val_ratio)

if config.verbose:
    for term in [train_dataset,val_dataset,test_dataset]:
        print(term[0].shape,term[1].shape)
    # show_images(*val_dataset,dataset_shape)

# Train Model

model = get_model(config,model_config)

train_model(config,model,train_dataset,val_dataset)

