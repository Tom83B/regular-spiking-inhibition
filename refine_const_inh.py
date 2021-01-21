"""Script used for eventual refinement of results produced by
HH/obtain_HH_heatmap.py
LIF/obtain_LIF_heatmap.py

this script refines simulations where the pre-synaptic inhibitory activity is varied"""

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

from matsim import *

from utils import SpaceScan, linear_roots

tott = 10000

def run_lif(exc_rate, inh_rate, seed, glif_type='dthr', gleak=0.045, tott=tott, dt=0.025):
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
    if glif_type == 'dthr':
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
        dt = 0.025 * 100 / exc_rate
#         tott = tott * 100 / exc_rate

    exc = ShotNoiseConductance(
        rate=exc_rate,
        g_peak=0.0015,
        reversal=0,
        decay=3)

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

def run_hh(exc_rate, inh_rate, seed, adapt=0, VS=-10, Ah=0.128, tott=tott, apcut=-25, progress=False, reverse=False):
    neuron = HHNeuron(adaptation=adapt, VS=VS, Ah=Ah)
    
    dt = 0.025
    
    if exc_rate > 100:
        dt = 0.025 * 100 / exc_rate
        tott = tott * 100 / exc_rate

    exc = OUConductance(
        rate=exc_rate,
        g_peak=0.0015,
        reversal=0,
        decay=3)

    inh = OUConductance(
        rate=inh_rate,
        g_peak=0.0015,
        reversal=-75,
        decay=10)

    neuron.append_conductance(exc)
    neuron.append_conductance(inh)
    
    v_arr_all = []
    v_arr = deque()
    v_mean_arr = deque()
    M2_arr = deque()
    spike_control = deque([-100,-100,-100])

    spike_times = []

    refrac = 0

    t = 0
    n = 0

    v_mean = 0
    M2 = 0
    
    if progress == True:
        mtqdm = tqdm
    else:
        mtqdm = lambda x: x
    
    for i in mtqdm(range(int(tott * 1000 / dt))):
        t += dt
        neuron.timestep(dt)
        refrac -= dt

        spike_control.append(neuron.voltage)
        spike_control.popleft()

        if spike_control[1] > apcut and spike_control[1] > spike_control[0] and spike_control[1] > spike_control[2]:
            spike_times.append(t)

    return np.diff(spike_times)

hh_data_root = 'HH experiments/data/'
hh_res = {}
hh_names = ['clas','musc','dynt']

for name in hh_names:
    filename = hh_data_root + f'res_{name}_area_inh.pickle'

    with open(filename, 'rb') as file:
        hh_res[name] = pickle.load(file)
        
lif_data_root = 'LIF/data/'
lif_res = {}
lif_names = ['dlif']

for name in lif_names:
    filename = lif_data_root + f'res_{name}_area_inh.pickle'

    with open(filename, 'rb') as file:
        lif_res[name] = pickle.load(file)
        
neuron_types = {
    'classical': {'adapt': 0, 'VS': -10, 'Ah': 0.128},
    'mcurrent':  {'adapt': 0.5, 'VS': -10, 'Ah': 0.128},
    'dynthresh': {'adapt': 0, 'VS': 8+6, 'Ah': 0.00128}
}

def run_wrapper(eid, *args):
    return (eid, *args, run_lif(*args))

def run_wrapper_hh(eid, *args):
    return (eid, *args, run_hh(*args, **neuron_types['dynthresh']))

def outputs2res(outputs):
    col_names = ['eid','exc','inh','seed','isis']

    res = {}

    for ix, sub_df in pd.DataFrame(outputs, columns=col_names).groupby(['eid','exc','inh']):
        eid, exc, inh = ix
        gr = res.get(inh, [])

        joined = np.concatenate(sub_df['isis'].values)
        mu, std = joined.mean(), joined.std()
        gr.append([exc, 1000/mu, std/mu])

        gr.sort(key=lambda x: x[0])
        res[inh] = gr
    
    for inh in res.keys():
        res[inh] = np.array(res[inh])
    
    return res


def get_outputs(mtype):
    if mtype == 'lif':
        scan = SpaceScan(lif_res['dlif'], q=1)
    elif mtype == 'hh':
        scan = SpaceScan(hh_res['dynt'], q=1)
    
    inputs = []
    outputs = []

    for inh in [0, 10, 20, 30, 40, 50]:
        exc_arr = scan.sample_contour(inh)
        rate_arr = scan.exc_rate_interpolation(exc_arr, inh)


        for eid, (e, r) in enumerate(zip(exc_arr, rate_arr)):
            nspikes = r * 1000
            nseg = int(np.ceil(10000 / nspikes))

            inputs.extend([(eid, e, inh)] * nseg)

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
    res_lif = get_outputs('lif')

    filename = 'data/dlif_refined_inh.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(res_lif, handle)
        
    res_hh = get_outputs('hh')

    filename = 'data/hhdt_refined_inh.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(res_hh, handle)