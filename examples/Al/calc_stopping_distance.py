#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(f"/scratch/yifany6/stopping-power-ml")
from stopping_power_ml.rc import *
from stopping_power_ml.cell import Cell
from stopping_power_ml.stopping.stopping_distance import compute_stopping_distance
import pickle as pkl
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm
import h5py 
from datetime import datetime

mpath = f'./model.h5'
model = keras.models.load_model(mpath)
with open(f'./featurizers.pkl', 'rb') as fp:
    featurizers = pkl.load(fp)
print(featurizers.feature_labels())
start_frame = pkl.load(open(f"al_starting_frame.pkl", 'rb'))
model.summary()

cell = Cell(start_frame) 
super_cell_lattice = cell.simulation_cell_lattice()[2, 2]
unit_cell_lattice = cell.conventional_cell_lattice()[2, 2]
print(super_cell_lattice, unit_cell_lattice)

cart_pos = np.array([1.92300000, 1.92300000, -7.69200000])
position = cell.cartesian_to_conventional(cart_pos)

dire = [0, 0, 1]
dirname = 'hyperchannel'

start_pos = cart_pos # np.dot(cell.simulation_cell_lattice(), position)
vdir = np.array(dire, dtype = float)
vdir *= 1 / np.linalg.norm(vdir)

v = np.linspace(0.8, 4, 8, endpoint = True)

for i, vi in enumerate(tqdm(v)):
    sd, output = compute_stopping_distance(cell, model, featurizers, position, vdir*vi, output = True) 
    print("\nstopping distance", sd)
    plt.plot(output['displacement'], output['velocity'])

plt.show()

