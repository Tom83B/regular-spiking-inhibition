"""Script used for estimating rate-CV dependence for different values of excitation-to-inhibition
ratio or for different values of inhibitory pre-synaptic activity

The script takes the following command line arguments:
-c/--clas, -m/--musc, -d/--dynt : these flags specify which models should be simulated
    clas: HH model without SFA
    musc: HH model with M-current SFA
    dynt: HH model with dynamic threshold
-a/--append : whether results should be appended to an already existing file
--fspace : specifies which values of e-i ratio or inhib. activity should be simulated
    3 numbers, if "--fspace 0 0.9 10" the tested values will be np.linspace(0, 0.9, 20)
-s/--scan : if "--scan f" the values specified in --fspace correspond to the ratio of e(kHz)/i(kHz)
            if "--scan inh" the values specified in --fspace represent the rate of inhibitory pre-synaptic activity (kHz)
--postfix : the results are saved in res_{clas/musc/dynt}_{postfix}.pickle"""

import numpy as np
from tqdm import tqdm
import json
from scipy.interpolate import griddata
from collections import deque
from itertools import product
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splprep, splev
from functools import partial
from test import square, tmp_func
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import norm
import argparse
import pickle
from matsim import *

def run(f, intensity, adapt=0, VS=-10, Ah=0.128, tott=1, apcut=-25, progress=True, reverse=False):
    neuron = HHNeuron(adaptation=adapt, VS=VS, Ah=Ah)

    if reverse == False:
        exc_rate, inh_rate = intensity, f * intensity
    else:
        inh_rate, exc_rate = intensity, f * intensity
    
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

    return spike_times

def get_rate(f, intensity, tott=100, **kwargs):
    spiketrain = run(f,intensity, tott, **kwargs)
    return len(spiketrain) / tott

def get_stats(f, intensity, adapt, VS, Ah, tott=30001, **kwargs):
    spiketrain = np.array(run(f,intensity, adapt, VS, Ah, tott, **kwargs))
    spiketrain = spiketrain[spiketrain > 1000]
    if len(spiketrain) > 10:
        isis = np.diff(spiketrain)
        return 1000 / isis.mean(), isis.std() / isis.mean()
    else:
        return len(spiketrain) / tott, np.nan
    
def get_stats_rates(exc_rate, inh_rate, adapt, VS, Ah, tott=1000, **kwargs):
    f = inh_rate / exc_rate
    return get_stats(f, exc_rate, adapt, VS, Ah, tott, **kwargs)

neuron_types = {
    'classical': {'adapt': 0, 'VS': -10, 'Ah': 0.128},
    'mcurrent':  {'adapt': 0.5, 'VS': -10, 'Ah': 0.128},
    'dynthresh': {'adapt': 0, 'VS': 8+6, 'Ah': 0.00128}
}
    
def scan_space(neuron_type, fspace, scan_type):
    stop_mark = False
    res = {}
    
    adapt = neuron_types[neuron_type]['adapt']
    VS = neuron_types[neuron_type]['VS']
    Ah = neuron_types[neuron_type]['Ah']
    
#     for f in tqdm(np.linspace(0., 0.9, 20)):
    for f in tqdm(np.linspace(fspace[0], fspace[1], int(fspace[2]))):
        stats = []

        xx = np.linspace(-1, np.log10(140), 500)
        yy = np.zeros_like(xx)
        exc_rates = np.logspace(0, 3, 36)
        
        if not stop_mark:
            for n in tqdm(range(2)):
                adapt_arr = np.ones_like(exc_rates) * adapt
                VS_arr = np.ones_like(exc_rates) * VS
                Ah_arr = np.ones_like(exc_rates) * Ah

                if scan_type == 'f':
                    new_stats = Pool(36).map(get_stats, np.ones_like(exc_rates) * f, exc_rates, adapt_arr, VS_arr, Ah_arr)
                elif scan_type == 'inh':
                    new_stats = Pool(36).map(get_stats_rates, exc_rates, np.ones_like(exc_rates) * f, adapt_arr, VS_arr, Ah_arr)
                new_stats = [(rate, x[0], x[1]) for rate, x in zip(exc_rates, new_stats)]

                stats.extend(new_stats)
                stats.sort(key=lambda tup: tup[0])

                intensities = [x[0] for x in stats]
                rates = np.array([x[1] for x in stats])

                if (rates > 1).sum() == 0:
                    stop_mark = True
                    break


                def get_exc_rates(rate):
                    func_rate = UnivariateSpline(np.log10(intensities), rates-rate, s=0)
                    log_exc_rates = func_rate.roots()
                    return 10**log_exc_rates

                for rate in rates[(rates > 1) & (rates < 120)]:
                    exc_rates = get_exc_rates(rate)
                    if len(exc_rates) > 0:
                        yy += norm.pdf(xx, loc=np.log10(rate), scale=0.1) / len(exc_rates)

                yy_tmp = yy.copy()

                new_exc_rates = []

                for ii in range(10000):
                    new_log_rate = xx[np.argmin(yy_tmp)]
                    new_rate = 10**new_log_rate
                    ers = get_exc_rates(new_rate)

                    if new_rate > 1 and new_rate < 120:
                        for er in ers:
                            if er not in new_exc_rates and er not in intensities:
                                new_exc_rates.append(er)

                    if len(ers) > 0:
                        yy_tmp += norm.pdf(xx, loc=new_log_rate, scale=0.1) / len(ers)
                    else:
                        yy_tmp += norm.pdf(xx, loc=new_log_rate, scale=0.1)
                    
                    if len(new_exc_rates) >= 36:
                        break
                else:
                    stop_mark = True
                    break

                exc_rates = new_exc_rates[:36]
            res[f] = np.array(stats)
    return res

def save_data(res, filename, append=False):
    if append:
        with open(filename, 'rb') as handle:
            tmp = pickle.load(handle)
        
        for f, res_grid in res.items():
            tmp[f] = res_grid
            
        with open(filename, 'wb') as handle:
            pickle.dump(tmp, handle)
    else:
        with open(filename, 'wb') as handle:
            pickle.dump(res, handle)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c','--clas', action='store_true')
    parser.add_argument('-m','--musc', action='store_true')
    parser.add_argument('-d','--dynt', action='store_true')
    parser.add_argument('-a','--append', action='store_true')
    parser.add_argument('--fspace', type=float, nargs=3, default=[0, 0.9, 20])
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('-s','--scan', type=str, default='f')
    
    
    args = parser.parse_args()
    
    if args.clas:
        print('CLASSICAL')
        res_clas = scan_space('classical', args.fspace, args.scan)
        filename = f'data/res_clas_{args.postfix}.pickle'
        save_data(res_clas, filename, args.append)
#         with open('data/res_hh_clas.pickle', 'wb') as handle:
#             pickle.dump(res_clas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if args.musc:
        print('M-CURRENT')
        res_musc = scan_space('mcurrent', args.fspace, args.scan)
        filename = f'data/res_musc_{args.postfix}.pickle'
        save_data(res_musc, filename, args.append)
#         with open('data/res_hh_musc.pickle', 'wb') as handle:
#             pickle.dump(res_musc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if args.dynt:
        print('DYNAMIC THRESHOLD')
        res_dynt = scan_space('dynthresh', args.fspace, args.scan)
        filename = f'data/res_dynt_{args.postfix}.pickle'
        save_data(res_dynt, filename, args.append)
#         with open('data/res_hh_dynt.pickle', 'wb') as handle:
#             pickle.dump(res_dynt, handle, protocol=pickle.HIGHEST_PROTOCOL)