"""Various plotting, interpolating and statistical utilities"""

import numpy as np
import functools
from scipy.interpolate import UnivariateSpline, griddata
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class StatsTracker:
    """Online computation of mean and standard deviation"""
    def __init__(self):
        self.n = 0
        self.average = None
        self.M2 = None
        self.var = None
    
    def add(self, x):
        if self.n == 0:
            self.average = x
            self.M2 = x * 0
            self.var = x * 0
            self.n += 1
            self.var = self.M2 / self.n
        else:
            new_average = (self.n*self.average + x) / (self.n+1)
            self.M2 += (x - self.average) * (x - new_average)
            self.n += 1
            self.var = self.M2 / self.n
            self.average = new_average
    
    def std(self):
        return np.sqrt(self.var)

def linear_roots(xx, yy):
    """Find roots of a function by linear interpolation"""
    xx = np.array(xx)
    yy = np.array(yy)
    
    root_ixs = np.argwhere((yy[:-1] * yy[1:]) < 0).flatten()
    
    roots = []
    
    for ix in root_ixs:
        t1 = np.abs(yy[ix])
        t2 = np.abs(yy[ix+1])
        roots.append((t2*xx[ix] + t1*xx[ix+1]) / (t1+t2))
    
    return np.array(roots)

def min_c(rate):
    T = 1/rate
    dtheta = 4
    ttheta = 0.1
    th0 = -55
    Ee = 0
    Ei = -75

    exp_factor = 1 - np.exp(T/ttheta)
    vinf = (th0 * exp_factor - dtheta) / exp_factor
#     vinf = 4*(np.exp(T/ttheta)-1)+th0
    min_c = (Ee - vinf) / (vinf - Ei)

    return min_c

def psfr_scaler(psfr):
    return np.log10(psfr) / 2

def cv_scaler(cv):
    return cv / 0.8

class SpaceScan:
    """Class for working with simulation results"""
    def __init__(self, res, fill=False, q=10/3, dlif=False):
        self.dlif = dlif
        self.res = {}
        self.q = q
        for f, grid_res in list(res.items()):
            grid_res_new = []
            for row in grid_res:
                if row[1] < 150:
                    grid_res_new.append(row)
                else:
                    break
            self.res[np.round(f,5)] = np.array(grid_res_new)
        
        pairs = []
        for f, grid_res in res.items():
            pairs.append([f*self.q, res[f][:,1].max()])

        pairs.sort(key=lambda x: x[0])
        self._max_func = UnivariateSpline(*np.array(pairs).T, s=0, k=1)
        
        pairs_exc = []
        for f, grid_res in res.items():
            ix = np.argwhere(grid_res[:,2] == grid_res[:,2]).flatten()[-1]
            pairs_exc.append([f*self.q, np.log10(grid_res[:,0][ix])])
        
        pairs_exc.sort(key=lambda x: x[0])
        self._max_exc_func = UnivariateSpline(*np.array(pairs_exc).T, s=0, k=1)
#         self._max_exc_func = lambda x: 1e8        
    
        pairs_exc = []
        for f, grid_res in res.items():
            ix = np.argwhere(grid_res[:,1] > 0.0).flatten()[0]
            pairs_exc.append([f*self.q, np.log10(grid_res[:,0][ix])])
        
        pairs_exc.sort(key=lambda x: x[0])
        self._min_exc_func = UnivariateSpline(*np.array(pairs_exc).T, s=0, k=1)
        
        self.fill = fill
#         if fill:
            

#             xx = np.logspace(0, 5, 100)

#             self.res_filled = {}
#             for f in np.linspace(0, 1, 100):
#                 points = np.array([[np.log10(x), f*10/3] for x in xx])
#                 rates = griddata((np.log10(x), y), z1, points, method='linear', rescale=True)
#                 cvs = griddata((np.log10(x), y), z2, points, method='linear', rescale=True)
#                 self.res_filled[f] = np.array([xx, rates, cvs]).T
    
    def include_limits(self):
        dtheta = 4
        ttheta = 0.1
        th0 = -55
        Ee = 0
        Ei = -75

        for key, grid_res in self.res.items():
            vinf = (Ee+key*self.q*Ei) / (1+key*self.q)
            if vinf > th0:
                T = ttheta * np.log(1 + dtheta/(vinf-th0))
                r = 1 / T
                row = np.array([1e6, r, 0])
                self.res[key] = np.append(grid_res, row).reshape((-1, 3))
        
        
        pairs = []
        for f, grid_res in self.res.items():
            pairs.append([f*self.q, grid_res[:,1].max()])

        pairs.sort(key=lambda x: x[0])
        self._max_func = UnivariateSpline(*np.array(pairs).T, s=0, k=1)
#             else:
#                 row = np.array([1e6, 0.1, 1])
#                 self.res[key] = np.append(grid_res, row).reshape((-1, 3))
    
    @functools.cached_property
    def _gridvals_f(self):
            
        
            x = []
            y = []
            z1 = []
            z2 = []

            for f, grid_res in list(self.res.items()):
                mask = grid_res[:,2] == grid_res[:,2]
#                 f_rate = UnivariateSpline(np.log10(grid_res[:,0]), np.log10(grid_res[:,1]), k=1, s=0)
#                 f_std = UnivariateSpline(np.log10(grid_res[:,0]), grid_res[:,2], k=1, s=0, ext='extrapolate')
                
#                 xx = np.logspace(0, 6, 500)
#                 ff = [f*self.q] * len(xx)
#                 zz1 = f_rate(np.log10(xx))
#                 zz2 = f_std(np.log10(xx))
#                 x.extend(list(xx))
#                 y.extend(list(ff))
#                 z1.extend(list(zz1))
#                 z2.extend(list(zz2))
                for row in grid_res[mask]:
                    x.append(row[0])
                    y.append(f*self.q)
                    z1.append(row[1])
                    z2.append(row[2])
            
            return (x, y), z1, z2
    
    def exc_rate_interpolation(self, exc, c):
        (x, y), z1, z2 = self._gridvals_f
        points = np.array([[np.log10(x), c] for x in exc])
        rates = griddata((np.log10(x), y), z1, points, method='linear', rescale=True)
        
        for i, (e, c) in enumerate(points):
            if e > self._max_exc_func(c):
                rates[i] = np.nan
            if e < self._min_exc_func(c):
                rates[i] = np.nan
        return rates
    
    def rate_exc_interpolation(self, rate, c):
        (x, y), z1, z2 = self._gridvals_f
        exc = np.logspace(self._min_exc_func(c),5,5000)
        points = np.array([[np.log10(x), c] for x in exc])
        rates = self.exc_rate_interpolation(exc, c)
#         rates = griddata((np.log10(x), y), z1, points, method='linear', rescale=True)
        
        mask = (rates == rates)
        return linear_roots(exc[mask], rates[mask]-rate)
#         func = UnivariateSpline(exc[mask], rates[mask]-rate, k=3, s=0)
#         return func.roots()
    
    def exc_cv_interpolation(self, exc, c):
        (x, y), z1, z2 = self._gridvals_f
        points = np.array([[np.log10(x), c] for x in exc])
        cvs = griddata((np.log10(x), y), z2, points, method='linear', rescale=True)
        return cvs
    
    def sample_contour(self, c, exc_start=None, shift_factor=1.1, point_dist=0.1, max_cv=0.9):
        if exc_start is None:
            try:
                exc_start = self.rate_exc_interpolation(0.8, c=c)[0]
            except:
                out_rates = np.linspace(0.8, 1.5, 21)
                for o_r in out_rates:
                    try:
                        exc_start = self.rate_exc_interpolation(o_r, c=c)[0]
                        break
                    except:
                        continue
                else:
                    return []
#                     import pdb; pdb.set_trace()
            
        exc_arr = [exc_start]

        for i in range(100):
            rate_start = self.exc_rate_interpolation([exc_start], c=c)[0]
            if rate_start != rate_start:
                exc_arr = exc_arr[:-1]
                break
            elif rate_start > 120:
                break
            cv_start = self.exc_cv_interpolation([exc_start], c=c)[0]
            if cv_start > max_cv:
                break

            exc_end = exc_start * shift_factor

            exc_tmp = np.logspace(np.log10(exc_start), np.log10(exc_end), 100)
            log_scaled_rates = psfr_scaler(self.exc_rate_interpolation(exc_tmp, c))
            scaled_cvs = cv_scaler(self.exc_cv_interpolation(exc_tmp, c))
            dists = np.sqrt((log_scaled_rates-psfr_scaler(rate_start))**2 + (scaled_cvs-cv_scaler(cv_start))**2)

            try:
                exc_start = 10**linear_roots(np.log10(exc_tmp), dists - point_dist)[0]
            except:
                exc_start = exc_tmp[np.argmax(dists)]
            exc_arr.append(exc_start)
        
        return exc_arr
        
    
    @functools.cached_property
    def res_filled(self):
        tmp = {}
        xx = np.logspace(0, 6, 200)
        for f in np.linspace(0, max(list(self.res.keys())), 100):
#             points = np.array([[x, f*10/3] for x in xx])
            rates = self.exc_rate_interpolation(xx, f*self.q)
            cvs = self.exc_cv_interpolation(xx, f*self.q)
            tmp[f] = np.array([xx, rates, cvs]).T
        return tmp

    @functools.cached_property
    def _gridvals(self):
        if self.fill:
            res = self.res_filled
        else:
            res = self.res
        
        x = []
        y = []
        z = []
        
        if self.dlif:
            x2 = []
            y2 = []
            z2 = []
            
            for rate in np.logspace(0, 2, 100):
                T = 1/rate
                dtheta = 4
                ttheta = 0.1
                th0 = -55
                Ee = 0
                Ei = -75

                exp_factor = 1 - np.exp(T/ttheta)
                vinf = (th0 * exp_factor - dtheta) / exp_factor
                c = (Ee - vinf) / (vinf - Ei)
                x2.append(rate)
                y2.append(c)
                z2.append(0)

        for f, grid_res in list(res.items())[::]:
            mask = (grid_res[:,1] == grid_res[:,1]) & (grid_res[:,2] == grid_res[:,2])
            
            if len(grid_res[mask]) > 1:
                func_stdev = UnivariateSpline(grid_res[mask,0], grid_res[mask,2], s=0, k=1)

                rate_arr = np.logspace(0,2,50)
#                 rate_arr = grid_res[:,1]
#                 print(grid_res[:,1])
#                 rate_arr = grid_res[mask,1]

                stdev_arr = []
                excrate_arr = []
        
                rate_arr2 = []
                stdev_arr2 = []
                excrate_arr2 = []

                for rate in rate_arr:
#                     func_rate = UnivariateSpline(np.log10(grid_res[mask,0]), grid_res[mask,1]-rate, s=0)
#                     log_exc_rates = func_rate.roots()
                    log_exc_rates = linear_roots(np.log10(grid_res[mask,0]), grid_res[mask,1]-rate)
#                     log_exc_rates = log_exc_rates[log_exc_rates < max_log_exc]
                    if len(log_exc_rates) == 0:
                        break
                
                    if not self.dlif:
                        stdevs = func_stdev(10**log_exc_rates)
                        stdev_arr.append(stdevs.min())
                        excrate_arr.append(10**log_exc_rates[np.argmin(stdevs)])
                    else:
                        stdevs = func_stdev(10**log_exc_rates)
                        stdev_arr.append(stdevs[0])
                        excrate_arr.append(10**log_exc_rates[0]) 
                        
                        if len(log_exc_rates) > 1:
                            c = f*self.q
                            if c > min_c(rate):
                                stdevs = func_stdev(10**log_exc_rates)
                                stdev_arr2.append(stdevs[-1])
                                excrate_arr2.append(10**log_exc_rates[-1])
                                rate_arr2.append(rate)
                
                n = len(stdev_arr)
                x.extend(list(rate_arr[:n]))
                y.extend([f*self.q]*n)
                z.extend(stdev_arr)
            
                if self.dlif:
                    n2 = len(stdev_arr2)
                    x2.extend(rate_arr2)
                    y2.extend([f*self.q]*n2)
                    z2.extend(stdev_arr2)
            else:
                x.append(grid_res[0,1])
                y.append(f*self.q)
                z.append(grid_res[0,2])
        if self.dlif:
            return (x, y), z, (x2, y2), z2
        else:
            return (x, y), z
    
    def interpolate_cv(self, points):
        (x, y), z, *_ = self._gridvals
        x = np.log10(x)
        points[:,0] = np.log10(points[:,0])
        try:
            z = griddata((x, y), z, points, method='linear', rescale=True)
        except:
            import pdb;pdb.set_trace()
        
        for i, (x, y) in enumerate(points):
            if x > self._max_func(y):
                z[i] = np.nan
        
        return z
        
    def plot_heatmap(self, ax, vlim=None):
        if self.dlif:
            (x, y), z, (x2, y2), z2 = self._gridvals
            xi = np.logspace(0,2,200)
            yi = np.linspace(0, 3.2, 200)

            pairs = []
            for _x in xi:
                for _y in yi:
                    pairs.append([np.log10(_x),_y])
            pairs = np.array(pairs)

            zi = griddata((np.log10(x), y), z, pairs, method='linear', rescale=True, fill_value=0).reshape((len(xi), len(yi))).T
            zi2 = griddata((np.log10(x2), y2), z2, pairs, method='linear', rescale=True).reshape((len(xi), len(yi))).T

            for i in range(len(xi)):
                rate = xi[i]
                for j in range(len(yi)):
                    c = yi[j]
                    if c > min_c(rate):
                        zi[j,i] = 0
                    if rate > self._max_func(c):
                        zi[j,i] = np.nan
            
            ax.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
            CS = ax.contourf(xi,yi,zi,15,cmap=plt.cm.YlGn)
            ax.contour(xi,yi,zi2,CS.cvalues,linewidths=0.5,colors='k')
            CS = ax.contourf(xi,yi,zi2,CS.cvalues,cmap=plt.cm.YlGn)
        else:
            xi = []
            yi = []

            xi = np.logspace(0,2,200)
            yi = np.linspace(0, max(list(self.res.keys())) * self.q, 200)

            pairs = []
            for x in xi:
                for y in yi:
                    pairs.append([x,y])
            pairs = np.array(pairs)
            zi = self.interpolate_cv(pairs).reshape((len(xi), len(yi))).T

            CS = ax.contour(xi,yi,zi,10,linewidths=0.5,colors='k')
            if vlim is None:
                CS = ax.contourf(xi,yi,zi,10,cmap=plt.cm.YlGn)
            else:
                CS = ax.contourf(xi,yi,zi,10,cmap=plt.cm.YlGn, vmin=vlim[0], vmax=vlim[1])

        ax.set_xlabel('PSFR (Hz)')
        ax.set_xlim(1, 100)
        ax.set_xscale('log')

        return CS
    
    def plot_line(self, f, ax, s=0, label=None, **kwargs):
        grid_res = self.res[f]
        x = grid_res[:,1]
        y = grid_res[:,2]
        u = grid_res[:,0]
        
        mask = (x > 0) & (y == y)
        x = x[mask]
        y = y[mask]
        u = u[mask]
        
#         u1 = u[x < 10][::3]
#         y1 = y[x < 10][::3]
#         x1 = x[x < 10][::3]
#         u2 = u[x >= 10]
#         y2 = y[x >= 10]
#         x2 = x[x >= 10]
        
#         u = np.concatenate([u1,u2])
#         x = np.concatenate([x1,x2])
#         y = np.concatenate([y1,y2])
        

        frate = UnivariateSpline(u, x, s=0.00, k=1)
        fcv = UnivariateSpline(u, y, s=0.00, k=1)
        
        mod_u = np.sort(np.concatenate(([99.9, 100.1], u)))
        x = frate(mod_u)
        y = fcv(mod_u)
        
#         mask = (xx > 0) & (yy == yy)
#         tck_input, u_input = splprep([xx[mask], grid_res[mask,0]], s=0)
#         tck, u = splprep([xx[mask], yy[mask]], s=s)
        
        mask = mod_u <= 100
        ax.plot(x[mask], y[mask], **kwargs, label=label)
        ax.plot(x[~mask], y[~mask], linestyle='dotted', **kwargs)
    
    def separating_line(self, start_rate=5, cs_window=31):
        
        
        roots = []
        
        for rate in np.logspace(np.log10(start_rate), 2, 200):
            c_arr = np.linspace(0, min_c(rate), 300)
            points = np.array([[rate, c] for c in c_arr])
            sigmas = self.interpolate_cv(points)
            mask = sigmas == sigmas
            n = 21
            deriv = savgol_filter(sigmas[mask], n, polyorder=1, deriv=1, mode='interp')
            linroots = linear_roots(c_arr, deriv)
            if len(linroots) >= 1:
                while len(linroots) > 2:
                    n += 2
                    deriv = savgol_filter(sigmas[mask], n, polyorder=1, deriv=1, mode='interp')
                    linroots = linear_roots(c_arr, deriv)
    #                 print(linroots)
                roots.append((rate, linroots.mean()))

        roots = np.array(roots)
        
        rates, cs = roots.T
        for i, c in enumerate(cs):
            if c != c:
                cs[i] = (cs[i-1] + cs[i+1]) / 2
#         print(rates, cs)
        cs_filtered = savgol_filter(cs, cs_window, 1)
        
        return rates, cs_filtered