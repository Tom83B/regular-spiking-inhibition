"""Script simulating HH model with a sigmoidal shape increase in pre-synaptic inhibitory activity
Used for the Graphical abstract

The script takes the following command line arguments:
-c/--clas, -m/--musc, -d/--dynt : these flags specify which models should be simulated
    clas: HH model without SFA
    musc: HH model with M-current SFA
    dynt: HH model with dynamic threshold
-t/--total : the number of trials to be simulated. The more trials, the smoother Fano-factor curve
-i/--inh : value to which the inhibitory pre-synaptic activity should increase (in kHz)
"""

import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json
from collections import deque
from scipy.interpolate import UnivariateSpline, griddata
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import matplotlib.gridspec as gridspec
import argparse
from matsim import *

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import StatsTracker, SpaceScan
    
exc_lag = 10

def start_stop_vals(inh_start, inh_end, out_rate, scan):
    exc_start = scan.rate_exc_interpolation([out_rate], inh_start)[0]
    exc_end = scan.rate_exc_interpolation([out_rate], inh_end)[0]
    
    return exc_start, inh_start, exc_end, inh_end


def get_counts(neuron_type, exc_start, inh_start, exc_end, inh_end, trial=False):
    neuron_types = {
    'classical': {'adaptation': 0, 'VS': -10, 'Ah': 0.128},
    'mcurrent':  {'adaptation': 0.5, 'VS': -10, 'Ah': 0.128},
    'dynthresh': {'adaptation': 0, 'VS': 8+6, 'Ah': 0.00128}
        }
    
    apcut = -25
    neuron = HHNeuron(**neuron_types[neuron_type])
    offset = 1000
    tott = 2000 + offset
    
    def rate_func(t, es, inhs, ee, inhe):
        exc_rate = (ee-es) * (1 - 1 / (1 + np.exp((t-(1000+exc_lag+offset)) / 50))) + es
        inh_rate = (inhe-inhs) * (1 - 1 / (1 + np.exp((t-(1000+offset)) / 50))) + inhs
        return exc_rate, inh_rate

    exc = OUConductance(
        rate=exc_start,
        g_peak=0.0015,
        reversal=0,
        decay=3)

    inh = OUConductance(
        rate=inh_start,
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
    dt = 0.025
    n = 0

    v_mean = 0
    M2 = 0
    
    tt = np.arange(0, tott, dt)
    exc_arr, inh_arr = rate_func(tt, exc_start, inh_start, exc_end, inh_end)
    
    
    for t, exc_rate, inh_rate in zip(tt, exc_arr, inh_arr):
        exc.set_rate(exc_rate)
        inh.set_rate(inh_rate)
        neuron.timestep(dt)
        refrac -= dt

        spike_control.append(neuron.voltage)
        spike_control.popleft()

        if spike_control[1] > apcut and spike_control[1] > spike_control[0] and spike_control[1] > spike_control[2]:
            spike_times.append(t)

        v_arr_all.append(neuron.voltage)
    
    if trial:
        return np.ones_like(v_arr_all).cumsum() * dt - dt, v_arr_all
    
    counts = []

    for tstart in np.linspace(offset-100, tott-100, 100):
        spike_count = ((spike_times > tstart) & (spike_times <= tstart + 100)).sum()
        counts.append(spike_count)
    
    return np.array(counts).astype(float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c','--clas', action='store_true')
    parser.add_argument('-m','--musc', action='store_true')
    parser.add_argument('-d','--dynt', action='store_true')
    parser.add_argument('-t', '--total', type=int, default=3600)
    parser.add_argument('-i', '--inh', type=float, default=40)
    
    args = parser.parse_args()
    total = args.total
    
    if args.clas:
        with open('data/res_clas_area_inh.pickle', 'rb') as file:
            res_clas = pickle.load(file)
            scan_clas = SpaceScan(res_clas, q=1)
        
        data_clas = {}
        clas_tracker = StatsTracker()
        iarr = start_stop_vals(0, args.inh, 10, scan_clas)
        tarr, varr = get_counts('classical', *iarr, trial=True)

        intensity_arrs = list(np.array([iarr] * total).T)

        for counts in tqdm(Pool().imap(get_counts, ['classical']*total, *intensity_arrs), total=total):
            clas_tracker.add(counts)

        data_clas['run'] = (tarr, varr)
        data_clas['average'] = clas_tracker.average
        data_clas['std'] = clas_tracker.std()

        with open('../data/HH_clas_abstract.pickle', 'wb') as handle:
            pickle.dump(data_clas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    if args.dynt:
        with open('data/res_dynt_area_inh.pickle', 'rb') as file:
            res_dynt = pickle.load(file)
            scan_dynt = SpaceScan(res_dynt, q=1)
            
        data_dynt = {}
        dynt_tracker = StatsTracker()
        iarr = start_stop_vals(0, args.inh, 10, scan_dynt)
        intensity_arrs = list(np.array([iarr] * total).T)
        tarr, varr = get_counts('dynthresh', *iarr, trial=True)

        for counts in tqdm(Pool().uimap(get_counts, ['dynthresh']*total, *intensity_arrs), total=total):
            dynt_tracker.add(counts)

        data_dynt['run'] = (tarr, varr)
        data_dynt['average'] = dynt_tracker.average
        data_dynt['std'] = dynt_tracker.std()

        with open('../data/HH_dynt_abstract.pickle', 'wb') as handle:
            pickle.dump(data_dynt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    if args.musc:
        with open('data/res_musc_area_inh.pickle', 'rb') as file:
            res_musc = pickle.load(file)
            scan_musc = SpaceScan(res_musc, q=1)
            
        data_musc = {}
        musc_tracker = StatsTracker()
        iarr = start_stop_vals(0, args.inh, 10, scan_musc)
        intensity_arrs = list(np.array([iarr] * total).T)
        tarr, varr = get_counts('mcurrent', *iarr, trial=True)

        for counts in tqdm(Pool().uimap(get_counts, ['mcurrent']*total, *intensity_arrs), total=total):
            musc_tracker.add(counts)

        data_musc['run'] = (tarr, varr)
        data_musc['average'] = musc_tracker.average
        data_musc['std'] = musc_tracker.std()

        with open('../data/HH_musc_abstract.pickle', 'wb') as handle:
            pickle.dump(data_musc, handle, protocol=pickle.HIGHEST_PROTOCOL)