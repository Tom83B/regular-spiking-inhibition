"""Script used for eventual refinement of results produced by
HH/obtain_HH_heatmap.py
LIF/obtain_LIF_heatmap.py

this script refines simulations where the e-i ratio is varied"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json
from scipy.interpolate import griddata
from collections import deque
from itertools import product
# from multiprocessing import Pool, cpu_count
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splprep, splev
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
import pdb
import functools
import pickle
import matplotlib.gridspec as gridspec
from PIL import Image
import copy
import argparse

from matsim import *

from utils import SpaceScan, linear_roots

def run_lif(exc_rate, inh_rate, seed, glif_type, gleak=0.045, tott=1000, dt=0.025):
    A = 34636*1e-8
    C = 1*1e3*A
    gL = gleak*1e3*A
    R = 1/gL
    EL = -80
    V_thr = -55
    Ee = 0
    Ei = -75
    V_r = EL
    
    a1 = 0
    if glif_type == 'dynt':
        a1 = 4
    
    RS = MATThresholds(
        alpha1=a1,
        alpha2=0,
        tau1=100,
        tau2=200,
        omega=V_thr,
        refractory_period=0,
        resetting=1,
        name='thr')

    neuron = Neuron(
        resting_potential=EL,
        membrane_resistance=R,
        membrane_capacitance=C,
        mats=[RS],
        reset_potential = EL
    )

#     exc_rate, inh_rate = intensity, f*intensity
    
    if exc_rate > 100:
        exc = OUConductance(
            rate=exc_rate,
            g_peak=0.0015,
            reversal=0,
            decay=3)
    else:
        exc = ShotNoiseConductance(
            rate=exc_rate,
            g_peak=0.0015,
            reversal=0,
            decay=3)
    
    if inh_rate > 100:
        inh = OUConductance(
            rate=inh_rate,
            g_peak=0.0015,
            reversal=-75,
            decay=10)
#         dt = 0.025 * 100 / exc_rate
#         tott = tott * 100 / exc_rate
    else:
        inh = ShotNoiseConductance(
            rate=inh_rate,
            g_peak=0.0015,
            reversal=-75,
            decay=10)

    exc.set_g(exc_rate*0.0015*3)
    inh.set_g(inh_rate*0.0015*10)

    neuron.append_conductance(exc)
    neuron.append_conductance(inh)

    if glif_type == 'musc':
        adapt = ExponentialConductance(
            g_peak=0.01,
            reversal=-100,
            decay=100
        )
        neuron.append_conductance(adapt)
    
    spike_times = steady_spike_train(neuron, (tott+1)*1000, dt, seed)
    spike_times = spike_times[spike_times > 1000]
    
    return np.diff(spike_times)

def run_hh(exc_rate, inh_rate, seed, adapt=0, VS=-10, Ah=0.128, tott=1000, apcut=-25, progress=False, reverse=False):
    neuron = HHNeuron(adaptation=adapt, VS=VS, Ah=Ah)
    
    dt = 0.025
    
    if exc_rate > 100:
        exc = OUConductance(
            rate=exc_rate,
            g_peak=0.0015,
            reversal=0,
            decay=3)
    else:
        exc = ShotNoiseConductance(
            rate=exc_rate,
            g_peak=0.0015,
            reversal=0,
            decay=3)
    
    if inh_rate > 100:
        inh = OUConductance(
            rate=inh_rate,
            g_peak=0.0015,
            reversal=-75,
            decay=10)
#         dt = 0.025 * 100 / exc_rate
#         tott = tott * 100 / exc_rate
    else:
        inh = ShotNoiseConductance(
            rate=inh_rate,
            g_peak=0.0015,
            reversal=-75,
            decay=10)

    neuron.append_conductance(exc)
    neuron.append_conductance(inh)
    
    spike_times = steady_spike_train_hh(neuron, (tott+1)*1000, dt, seed)
    spike_times = spike_times[spike_times > 1000]
    
    return np.diff(spike_times)

hh_data_root = 'HH/data/'
hh_res = {}
hh_names = ['clas','musc','dynt']

for name in hh_names:
    filename = hh_data_root + f'res_{name}_area.pickle'

    with open(filename, 'rb') as file:
        hh_res[name] = pickle.load(file)
        
lif_data_root = 'LIF/data/'
lif_res = {}
lif_names = ['lif','dlif','mlif']
lif_name_map = {
    'clas': 'lif',
    'musc': 'mlif',
    'dynt': 'dlif'
}

for name in lif_names:
    filename = lif_data_root + f'res_{name}_area.pickle'

    with open(filename, 'rb') as file:
        lif_res[name] = pickle.load(file)
        
neuron_types = {
    'clas': {'adapt': 0, 'VS': -10, 'Ah': 0.128},
    'musc':  {'adapt': 0.5, 'VS': -10, 'Ah': 0.128},
    'dynt': {'adapt': 0, 'VS': 8+6, 'Ah': 0.00128}
}

def outputs2res(outputs):
    col_names = ['eid','c','exc','isis']

    res = {}

    for ix, sub_df in pd.DataFrame(outputs, columns=col_names).groupby(['eid','c','exc']):
        eid, c, exc = ix
        gr = res.get(c, [])

        joined = np.concatenate(sub_df['isis'].values)
        mu, std = joined.mean(), joined.std()
        gr.append([exc, 1000/mu, std/mu])

        gr.sort(key=lambda x: x[0])
        res[c] = gr
    
    for inh in res.keys():
        res[c] = np.array(res[c])
    
    return res


def get_outputs(mtype, tott, pd, ntype, max_cv=0.9):
    if mtype == 'lif':
        scan = SpaceScan(lif_res[lif_name_map[ntype]])
    elif mtype == 'hh':
        scan = SpaceScan(hh_res[ntype])
    
    def run_wrapper(eid, *args):
        c, exc, seed = args
        f = c*3/10
        inh = f*exc
        return (eid, c, exc, run_lif(exc, inh, seed, tott=tott, glif_type=ntype))

    def run_wrapper_hh(eid, *args):
        c, exc, seed = args
        f = c*3/10
        inh = f*exc
        return (eid, c, exc, run_hh(exc, inh, seed, **neuron_types[ntype], tott=tott))
    
    inputs = []
    outputs = []
    
    if mtype == 'lif':
        sf = 2
    else:
        sf = 1.1

    for c in np.linspace(0, 3.2, 17):
        exc_arr = scan.sample_contour(c, shift_factor=2, point_dist=pd, max_cv=max_cv)
        rate_arr = scan.exc_rate_interpolation(exc_arr, c)


        for eid, (e, r) in enumerate(zip(exc_arr, rate_arr)):
            nspikes = r * 1000
            nseg = int(np.ceil(10000 / nspikes))

            inputs.extend([(eid, c, e)] * nseg)

    inputs = np.array(inputs)

    erates = np.array(inputs)[:,1]
    seeds = np.arange(len(erates))
    
    
    if mtype == 'lif':
        for x in tqdm(Pool().uimap(run_wrapper, *inputs.T, seeds), total=len(erates), smoothing=0):
            outputs.append(x)
    elif mtype == 'hh':
        for x in tqdm(Pool().uimap(run_wrapper_hh, *inputs.T, seeds), total=len(erates), smoothing=0):
            outputs.append(x)

    res_refined = outputs2res(outputs)
    return res_refined

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l','--lif', action='store_true')
    parser.add_argument('-hh', '--hh', action='store_true')
    parser.add_argument('-n', '--neuron', type=str, nargs=2, default=['hh', 'dynt'])
    parser.add_argument('-t', '--tott', type=int, default=1000)
    parser.add_argument('--maxcv', type=float, default=0.9)
    
    args = parser.parse_args()
    
    mtype, ntype = args.neuron
    
#     if args.lif:
#         print('LIF')
#         res_lif = get_outputs('lif', tott=args.tott, pd=0.02)

#         filename = 'data/dlif_refined.pickle'
#         with open(filename, 'wb') as handle:
#             pickle.dump(res_lif, handle)
    
    if mtype == 'lif':
        res_lif = get_outputs('lif', tott=args.tott, pd=0.05, ntype=ntype, max_cv=args.maxcv)

        filename = f'data/glif_{ntype}_refined.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(res_lif, handle)

    if mtype == 'hh':
        res_hh = get_outputs('hh', tott=args.tott, pd=0.05, ntype=ntype, max_cv=args.maxcv)

        filename = f'data/{mtype}_{ntype}_refined.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(res_hh, handle)