"""Script used for estimating rate-CV dependence for different values of excitation-to-inhibition
ratio or for different values of inhibitory pre-synaptic activity

The script takes the following command line arguments:
-l/--lif, -m/--musc, -d/--dynt : these flags specify which models should be simulated
    lif: simple LIF model
    musc: LIF model with M-current SFA
    dynt: LIF model with dynamic threshold
--gleak : the leak conductance (in muS/cm2)
--aifrac : Ai = aifrac*Ae, where Ai, Ae are the jumps in inh./exc. conductances
-a/--append : whether results should be appended to an already existing file
--fspace : specifies which values of e-i ratio or inhib. activity should be simulated
    3 numbers, if "--fspace 0 0.9 10" the tested values will be np.linspace(0, 0.9, 20)
-s/--scan : if "--scan f" the values specified in --fspace correspond to the ratio of e(kHz)/i(kHz)
            if "--scan inh" the values specified in --fspace represent the rate of inhibitory pre-synaptic activity (kHz)
--postfix : the results are saved in res_{lif/mlif/dlif}_{postfix}.pickle"""

import numpy as np
from tqdm import tqdm
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
import argparse
import pickle
from matsim import *

def run(f, intensity, glif_type, gleak, aifrac, tott, dt=0.025):
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

    exc_rate, inh_rate = intensity, f*intensity
    
    if exc_rate > 100:
        dt = 0.025 * 100 / exc_rate
        tott = tott * 100 / exc_rate

    exc = ShotNoiseConductance(
        rate=exc_rate,
        g_peak=0.0015,
        reversal=0,
        decay=3)

    inh = ShotNoiseConductance(
        rate=inh_rate/aifrac,
        g_peak=0.0015*aifrac,
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
    
#     print(exc_rate*inh_rate*1000*np.random.rand())
    seed = int(exc_rate*inh_rate*1000*np.random.rand() % 165168541)
    spike_times = steady_spike_train(neuron, tott*1000, dt, seed)#['thr']
    
    return spike_times

# def get_rate(f, intensity, gleak, tott=100, **kwargs):
#     spiketrain = run(f,intensity, tott, **kwargs)
#     return len(spiketrain) / tott

def get_stats(f, intensity, glif_type, gleak, aifrac, tott=50001, **kwargs):
    spiketrain = np.array(run(f,intensity, glif_type, gleak, aifrac, tott, **kwargs))
    spiketrain = spiketrain[spiketrain > 1000]
    
    if len(spiketrain) > 10:
        isis = np.diff(spiketrain)
        return 1000 / isis.mean(), isis.std() / isis.mean()
    else:
        return len(spiketrain) / tott, np.nan
    
def get_stats_rates(exc_rate, inh_rate, glif_type, gleak, aifrac, tott=1000, **kwargs):
    intensity = exc_rate
    f = inh_rate / exc_rate
    return get_stats(f, exc_rate, glif_type, gleak, aifrac, tott, **kwargs)

# neuron_types = {
#     'classical': {'adapt': 0, 'VS': -10, 'Ah': 0.128},
#     'mcurrent':  {'adapt': 0.5, 'VS': -10, 'Ah': 0.128},
#     'dynthresh': {'adapt': 0, 'VS': 8+6, 'Ah': 0.00128}
# }
    
def scan_space(glif_type, fspace, scan_type='f', gleak=0.045, aifrac=1):
    stop_mark = False
    res = {}
    
#     adapt = neuron_types[neuron_type]['adapt']
#     VS = neuron_types[neuron_type]['VS']
#     Ah = neuron_types[neuron_type]['Ah']
    
#     for f in tqdm(np.linspace(0., 0.9, 20)):
    for f in tqdm(np.linspace(fspace[0], fspace[1], int(fspace[2]))):
        stats = []

        xx = np.linspace(-1, np.log10(140), 500)
        yy = np.zeros_like(xx)
        
        exc_rates = np.logspace(-2, 5, 36)
        
        if not stop_mark:
            for n in tqdm(range(2)):
                if scan_type == 'f':
                    new_stats = Pool(36).map(get_stats, np.ones_like(exc_rates) * f, exc_rates, [glif_type]*len(exc_rates), [gleak]*len(exc_rates), [aifrac]*len(exc_rates))
#                     new_stats = [get_stats(f, exc_rate, glif_type, gleak, aifrac) for exc_rate in exc_rates]
                    
                elif scan_type == 'inh':
                    new_stats = Pool(36).map(get_stats_rates, exc_rates, np.ones_like(exc_rates) * f, [glif_type]*len(exc_rates), [gleak]*len(exc_rates), [aifrac]*len(exc_rates))
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

                for rate in rates[(rates > 0.5) & (rates < 120)]:
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
    
    parser.add_argument('-l','--lif', action='store_true')
    parser.add_argument('-m','--musc', action='store_true')
    parser.add_argument('-d','--dynt', action='store_true')
    parser.add_argument('--gleak', type=float, default=0.045)
    parser.add_argument('--aifrac', type=float, default=1)
    parser.add_argument('-a','--append', action='store_true')
    parser.add_argument('-s','--scan', type=str, default='f')
    parser.add_argument('--fspace', type=float, nargs=3, default=[0, 0.9, 19])
    parser.add_argument('--postfix', type=str, default='')
    
    
    args = parser.parse_args()
    
    
    if args.lif:
        print('LIF')
        res_clas = scan_space('lif', args.fspace, args.scan, args.gleak, args.aifrac)
        filename = f'data/res_lif_{args.postfix}.pickle'
        save_data(res_clas, filename, args.append)
#         with open('data/res_hh_clas.pickle', 'wb') as handle:
#             pickle.dump(res_clas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if args.musc:
        print('M-CURRENT')
        res_musc = scan_space('musc', args.fspace, args.scan, args.gleak, args.aifrac)
        filename = f'data/res_mlif_{args.postfix}.pickle'
        save_data(res_musc, filename, args.append)
#         with open('data/res_hh_musc.pickle', 'wb') as handle:
#             pickle.dump(res_musc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if args.dynt:
        print('DYNAMIC THRESHOLD')
        res_dynt = scan_space('dthr', args.fspace, args.scan, args.gleak, args.aifrac)
        filename = f'data/res_dlif_{args.postfix}.pickle'
        save_data(res_dynt, filename, args.append)
#         with open('data/res_hh_dynt.pickle', 'wb') as handle:
#             pickle.dump(res_dynt, handle, protocol=pickle.HIGHEST_PROTOCOL)