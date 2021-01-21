import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import poisson
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import argparse

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import StatsTracker
from matsim import OUConductance, ShotNoiseConductance, Neuron

def run_conductances(_):
    meanV = -50
    
    def lami_func(t):
        return 20*(1 - 1 / (1 + np.exp((t-1100) / 50)))
    
    gleak = 0.045
    A = 34636*1e-8
    C = 1*1e3*A
    gL = gleak*1e3*A
    R = 1/gL
    EL = -80
    V_thr = -55
    Ee = 0
    Ei = -75
    V_r = EL

    neuron = Neuron(
        resting_potential=EL,
        membrane_resistance=R,
        membrane_capacitance=C,
        mats=[]
    )

    exc = OUConductance(
        rate=2.67,
        g_peak=0.0015,
        reversal=0,
    #     reversal=1000,
        decay=3
    )

    inh = OUConductance(
        rate=3.73,
        g_peak=0.0015,
        reversal=-75,
    #     reversal=1000,
        decay=10
    )

    neuron.append_conductance(exc)
    neuron.append_conductance(inh)
    
    v = []
    ge_arr = []
    gi_arr = []

    T = 2100 # 5 -> 30
    t = 0
    dt = 0.01

    tt = np.arange(0, T, dt)
    i_shift = lami_func(tt-5)
    gi_shift = 0.0015 * 10 * i_shift
    ge = (gL*(EL-meanV) + gi_shift*(Ei-meanV)) / (meanV-Ee)
    exc_arr = ge / (0.0015*3)
    
    for t, i, e in zip(tt, lami_func(tt), exc_arr):
        exc.set_rate(e)
        inh.set_rate(i)
        t += dt
        v.append(neuron.voltage)
        ge_arr.append(exc.g)
        gi_arr.append(inh.g)
        
#         isyn = ge*(neuron.voltage-Ee) + gi*(neuron.voltage-Ei)
#             ge.append(exc.g)
#             gi.append(inh.g)
        neuron.timestep(dt)
    
    ge_arr = np.array(ge_arr)
    gi_arr = np.array(gi_arr)
    v_arr = np.array(v)
    isyn = ge_arr*(v_arr-Ee) + gi_arr*(v_arr-Ei)
    
    return tt, np.array(v), isyn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--total', type=int, default=3600)
    
    args = parser.parse_args()
    total = args.total

    conductance_tracker = StatsTracker()
    isyn_tracker = StatsTracker()

    for tt, varr, iarr in tqdm(Pool().uimap(run_conductances, np.ones(total)), total=total):
        conductance_tracker.add(varr)
        isyn_tracker.add(iarr)

    data_conductance = {}
    tarr, varr, isyn = run_conductances(0)
    data_conductance['run'] = (tarr, varr, isyn)
    data_conductance['average'] = conductance_tracker.average
    data_conductance['std'] = conductance_tracker.std()
    data_conductance['average_i'] = isyn_tracker.average
    data_conductance['std_i'] = isyn_tracker.std()

    with open('../data/voltage_conductance_abstract_inh_40kHz.pickle', 'wb') as handle:
        pickle.dump(data_conductance, handle, protocol=pickle.HIGHEST_PROTOCOL)