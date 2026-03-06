#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(f"/scratch/yifany6/stopping-power-ml")
from stopping_power_ml.rc import *
from stopping_power_ml.cell import Cell
from stopping_power_ml.stopping.stopping_power import compute_stopping_power
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

#position = np.random.rand(3) 
#position[2] = 0
#dire = [2, 1, 1]
#dirname = 'random211'

#position = np.random.rand(3) 
#dire = np.random.rand(3)
#dirname = 'random_dir'

os.makedirs(dirname, exist_ok = True)
name = f"{dirname}/{position[0]}{position[1]}{position[2]}"

start_pos = cart_pos # np.dot(cell.simulation_cell_lattice(), position)
vdir = np.array(dire, dtype = float)
vdir *= 1 / np.linalg.norm(vdir)

max_spacing = 0.001
v = np.linspace(0.000001, 4, 40, endpoint = True)

states = []

for i, vi in enumerate(tqdm(v)):
    sp = compute_stopping_power(cell, model, featurizers, position, vdir, vi, coordinate = 'conventional')[0]
    print('stopping power', sp)
    states.append([vi, sp])

states = np.vstack(states).reshape([-1, 2])

with h5py.File(f'{name}.h5', 'w') as h5f:
    # Create a dataset with compression
    dataset = h5f.create_dataset('states', data = states, compression = 'gzip', compression_opts = 4)
    # Add attributes
    info = {'start_pos': start_pos, 
            'vdir': vdir, 
            "model": mpath, 
            "shape": "n_vmag, 2",
            "col": "vmag, force",
            "timestamp": datetime.now().isoformat()}
    for i in info.keys():
        dataset.attrs[i] = info[i]

plt.plot(states[:, 0], states[:, 1], 'o')

plt.xlabel("Velocity (at. u.)")
plt.ylabel("Stopping power (at. u.)")

plt.show()
