import os, sys
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# constants
kB = 1.38064852e-23;
NAv = 6.0221409e23;

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate partial molar properties from simulation trajectory')
    parser.add_argument('path', type=str, help='Path to trajectory files, using \
                            fort.12 format: [box lengths (angstrom)] [energy (K)] [pressure (kPa)] [molecule numbers],\
                        if the path contains nested directory then each subdirectory represents one state point')
    parser.add_argument('-n', type=int, default=0, help='Number of independent simulations to load, use 0 to read all trajectories in [path].')
    parser.add_argument('-b', '--nbox', type=int, help='Number of simulation boxes', required=True)
    parser.add_argument('-p', '--pressure', default=0, type=float, help='Set pressure of NpT simulation in MPa')
    parser.add_argument('-i', '--interval', default=1, type=int, help='Keep every [i] cycles in the trajectory, i should\
            be a multuple of the pressure calculation interval to obtain the correct enthalpy.')
    parser.add_argument('--train', default=0.6, type=float, help='Fraction ot training samples in the trajectory.')
    args = parser.parse_args()
    return args



'''
Simulation trajectory class.
To be load from a text file with number of boxes specified
FILENAME could be one file or a list of files.
If is a list of files, then each file represents 
a trajectory at a different state point.
'''
class Trajectory:
    def __init__(self, filename, n_box, interval, setp=0):
        if type(filename) == list:
            self.n_states = len(filename)
            data_tmp = []
            for f in filename:
                data_tmp.append(np.loadtxt(f))
            # need to use the same number of frames in each state point
            # also the number of frames must be a multiple of interval 
            # to concatenate
            nframes = (min([x.shape[0] for x in data_tmp]) // n_box) // interval
            for i in range(len(data_tmp)):
                if (data_tmp[i].shape[0]) != nframes * n_box:
                    print("Warning: %d/%d lines used in file %s" 
                          % (nframes * n_box, data_tmp[i].shape[0], filename[i]))
                    data_tmp[i] = data_tmp[i][:nframes*n_box, :]
            data = np.vstack(data_tmp)
        else:
            data = np.loadtxt(filename)
            self.n_states = 1
        self.setp = setp
        
        # number of components
        self.n_comp = data.shape[1] - 5
        # use number of molecules to check whether n_box is correct:
        raw_n = data[:, -self.n_comp:]
        if np.var(raw_n[-n_box * n_box * 10::n_box, 0]) / np.mean(raw_n[-n_box * n_box * 10::n_box, :]) \
                > 0.5 * np.var(raw_n[-n_box * 10:, 0]) / np.mean(raw_n[-n_box * 10:, 0]):
            print("Warning: number of simulation boxes may be incorrect")
        # number of molecules
        self.n = raw_n.reshape(-1, n_box, self.n_comp).transpose(0, 2, 1)[::interval, :, :]
        # volume, angstrom^3
        self.v = (data[:, 0] * data[:, 1] * data[:, 2]).reshape(-1, 1, n_box)[::interval, :, :]
        # internal energy, K
        self.u = data[:, 3].reshape(-1, 1, n_box)[::interval, :, :]
        # pressure, kPa
        pressures = data[:, 4].reshape(-1, 1, n_box)[::interval, :, :]
        # use the box with less pressure fluctuation
        var_p = (np.var(pressures, axis=0) / np.mean(pressures, axis=0)).ravel()
        self.vaporbox = np.argmin(var_p)
        self.p = pressures[:, :, self.vaporbox]
        
    
    '''
    Returns numbers of molecules in box IBOX (1-indexed).
    '''
    def nmolec(self, ibox):
        return self.n[:, :, ibox - 1] 
    
    '''
    Returns molar fractions in box IBOX (1-indexed).
    '''
    def molfrac(self, ibox):
        return self.n[:, :, ibox - 1] / np.sum(self.n[:, :, ibox - 1], axis=1).reshape(-1, 1)
    
    '''
    Returns the volume of box IBOX in nm^3 (1-indexed).
    '''
    def vol(self, ibox):
        return self.v[:, :, ibox - 1] / 1000
    
    '''
    Returns the molecular volume (average volume for 1 molecule)
    of box IBOX in nm^3 (1-indexed).
    '''
    def molvol(self, ibox):
        return self.v[:, :, ibox - 1] / 1000 / np.sum(self.n[:, :, ibox - 1], axis=1).reshape(-1, 1)
    
    '''
    Returns the internal energy of box IBOX in K (1-indexed).
    '''
    def energy(self, ibox):
        return self.u[:, :, ibox - 1] 
    
    '''
    Returns the molecular internal energy of box IBOX in K (1-indexed).
    '''
    def molenergy(self, ibox):
        return self.u[:, :, ibox - 1] / np.sum(self.n[:, :, ibox - 1], axis=1).reshape(-1, 1)
    
    '''
    Returns the pressure of the system in kPa.
    Always returns the simulation pressure.
    '''
    def pressure(self):
        return self.p
    
    '''
    Returns the enthalpy of the system in K.
    Use the set pressure if the pressure is fixed, 
    otherwise use simulation pressure.
    '''
    def enthalpy(self, ibox):
        pressure = self.setp if self.setp > 0 else self.p
        return self.energy(ibox) + pressure * self.vol(ibox) * 1e-24 / kB
    
    '''
    Returns the molecular of the system in K.
    Use the set pressure if the pressure is fixed, 
    otherwise use simulation pressure.
    '''
    def molenthalpy(self, ibox):
        pressure = self.setp if self.setp > 0 else self.p
        return self.molenergy(ibox) + pressure * self.vol(ibox) * 1e-24 / kB / np.sum(self.n[:, :, ibox - 1], axis=1).reshape(-1, 1)
    

class PartialMolarPropertySolver:
    
    names = {'V': 'volume', 'U': 'energy', 'H': 'enthalpy'}
    
    def __init__(self, traj):
        self.traj = traj
    
    def get_data(self, target, ibox):
        if target == 'V':
            return traj.molfrac(ibox), traj.molvol(ibox)
        elif target == 'U':
            return traj.molfrac(ibox), traj.molenergy(ibox)
        elif target == 'H':
            return traj.molfrac(ibox), traj.molenthalpy(ibox)
        else:
            raise ValueError('Undefined property', target)
    
    def fit(self, x, y):
        raise NotImplemented
        
    def predict(self, x):
        raise NotImplemented
        
    def gradient(self, x):
        raise NotImplemented
        
    def get_partial(self, x):
        y_pred = self.predict(x)
        gradients = self.gradient(x)
        pmp = gradients + y_pred - np.sum(gradients * x, axis=1).reshape(-1, 1)
        if self.traj.n_states == 1:
            return np.mean(pmp, axis=0), np.std(pmp, axis=0), y_pred
        else:
            mean = []
            std = []
            nframes = x.shape[0] // self.traj.n_states
            for i in range(self.traj.n_states):
                mean.append(np.mean(pmp[i*nframes:(i+1)*nframes, :], axis=0))
                std.append(np.std(pmp[i*nframes:(i+1)*nframes, :], axis=0))
            return mean, std, y_pred
            
    
    def mad(self, weights, y_true, y_pred):
        # MAD in collective property, y_pred is molar property
        return np.mean(np.abs(y_pred - y_true)) * weights
    
    def solve(self, target, ibox, verbose=False):
        x, y = self.get_data(target, ibox)
        self.fit(x, y)
        ymean, ystd, y_pred = self.get_partial(x)
        if verbose:
            print("Partial molar %s for box %d" % (self.names[target], ibox))
            print("".join(["%f +/- %f\t" % (ymean[i], ystd[i]) for i in range(len(ymean))]))
        if self.traj.n_states == 1:
            mad = self.mad(np.sum(self.traj.nmolec(ibox), axis=1), y, y_pred)
        else:
            nframes = x.shape[0] // self.traj.n_states
            mad = []
            for i in range(self.traj.n_states):
                mad.append(self.mad(np.sum(
                    self.traj.nmolec(ibox)[i*nframes:(i+1)*nframes, :], axis=1),
                                   y[i*nframes:(i+1)*nframes, :], y_pred[i*nframes:(i+1)*nframes, :]))        
        return ymean, ystd, mad

    
class LinearSolver(PartialMolarPropertySolver):
    
    name = 'Linear Regression'
    
    def get_data(self, target, ibox):
        if target == 'V':
            return traj.nmolec(ibox), traj.vol(ibox)
        elif target == 'U':
            return traj.nmolec(ibox), traj.energy(ibox)
        elif target == 'H':
            return traj.nmolec(ibox), traj.enthalpy(ibox)
        else:
            raise ValueError('Undefined property', target)
    
    def fit(self, x, y):
        self.coeffs = (np.linalg.pinv(x) @ y)
        #print("%s coefficients:" % self.name, self.coeffs)
        
    def predict(self, x):
        return x @ self.coeffs
        
    def get_partial(self, x):
        if self.traj.n_states == 1:
            return self.coeffs.ravel(), np.zeros((x.shape[1])), self.predict(x)
        else:
            mean = [self.coeffs.ravel()] * self.traj.n_states
            std = np.zeros((x.shape[1])) * self.traj.n_states
            return mean, std, self.predict(x)
    
    def mad(self, weights, y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))
    
class BiasedLinearSolver(PartialMolarPropertySolver):
    
    name = 'Linear Regression with bias'
    
    def get_data(self, target, ibox):
        if target == 'V':
            return traj.nmolec(ibox), traj.vol(ibox)
        elif target == 'U':
            return traj.nmolec(ibox), traj.energy(ibox)
        elif target == 'H':
            return traj.nmolec(ibox), traj.enthalpy(ibox)
        else:
            raise ValueError('Undefined property', target)
    
    def fit(self, x, y):
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        self.coeffs = (np.linalg.pinv(x) @ y)
        #print("%s coefficients:" % self.name, self.coeffs)
        
    def predict(self, x):
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        return x @ self.coeffs
        
    def get_partial(self, x):
        if self.traj.n_states == 1:
            return self.coeffs.ravel(), np.zeros((x.shape[1])), self.predict(x)
        else:
            mean = [self.coeffs.ravel()] * self.traj.n_states
            std = np.zeros((x.shape[1])) * self.traj.n_states
            return mean, std, self.predict(x)
    
    def mad(self, weights, y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    
class QuadraticSolver(PartialMolarPropertySolver):
    
    name = 'Quadratic'
    # y = a + b@x + x@c@x.T + x@b
    
    def fit(self, x, y):
        self.x_fit = x
        self.y_fit = y
        self.featurizer = PolynomialFeatures(2, interaction_only=False)
        self.features = self.featurizer.fit_transform(self.x_fit)
        self.coeffs = (np.linalg.pinv(self.features) @ y)
        # get coefficients for each power
        # use naive for loop
        self.a = 0
        self.b = np.zeros((x.shape[1]))
        self.c = np.zeros((x.shape[1], x.shape[1]))
        for co, p in zip(self.coeffs.ravel(), self.featurizer.powers_):
            loc, = np.nonzero(p)
            if len(loc) > 0:
                loc = np.hstack([[m] * p[m] for m in loc])
            if len(loc) == 0:
                self.a = co
            elif len(loc) == 1:
                self.b[loc[0]] = co
            elif len(loc) == 2:
                self.c[loc[0], loc[1]] = co
            else:
                raise ValueError('Incorrect power!')
                
    def predict(self, x):
        return self.featurizer.transform(x) @ self.coeffs
    
    # dy/dx = b + x@(c+c.T)
    def gradient(self, x):
        return self.b.reshape(1, -1) + np.matmul(x, (self.c + self.c.T))          
        
    
class GPSolver(PartialMolarPropertySolver):
    
    name = 'Gaussian Process'
        
    def fit(self, x, y):
        self.kernel = 1 * kernels.RBF(length_scale=1.0)
        parameters = {'kernel': [self.kernel], 'alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4]}
        model = GaussianProcessRegressor(kernel=self.kernel, alpha=5e-4, random_state=0)
        self.cv = GridSearchCV(model, parameters, cv=5)
        self.norm = np.mean(y)
        self.cv.fit(x, y / self.norm)
        self.model = self.cv.best_estimator_
        self.x_fit = x
        self.y_fit = y
        
    def predict(self, x):
        return self.model.predict(x) * self.norm
    
    def gradient(self, x):
        alpha = self.model.get_params()['alpha']
        kernel = self.model.kernel_
        l = kernel.get_params()['k2__length_scale']
        K = kernel(x, self.x_fit)
        K_train = kernel(self.x_fit, self.x_fit)
        dX = self.x_fit.reshape(1, -1, self.x_fit.shape[1]) - x.reshape(-1, 1, x.shape[1])
        dX = dX.transpose(2, 0, 1)
        coeffs = np.matmul(np.linalg.inv(K_train + np.diag([alpha]*K_train.shape[0])), self.y_fit)
        dK = np.matmul(K * dX / (l ** 2), np.tile(coeffs, (self.x_fit.shape[1], 1, 1))).squeeze(-1).T
        return dK
    
if __name__ == '__main__':
    import json
    args = parse_args()
    #solvers = [LinearSolver, QuadraticSolver, GPSolver]
    solvers = [LinearSolver, BiasedLinearSolver, QuadraticSolver,]
    targets = ['V', 'U', 'H']
    #targets = ['V', 'U']
    verbose = False
    multiple_states = False
    
    files = []
    states = []
    for x in os.listdir(args.path):
        if os.path.isdir(os.path.join(args.path, x)) and x[0] != '.':
            multiple_states = True
            states.append(x)
            if args.n > 0:
                files.append([os.path.join(args.path, x, 'par%i.txt') % i for i in range(1, args.n + 1)])
            else:
                files.append([os.path.join(args.path, x, y) for y in os.listdir(aos.path.join(args.path, x)) if 'par' in y])
            
    if not multiple_states:          
        if args.n > 0:
            files = [os.path.join(args.path, 'par%i.txt') % i for i in range(1, args.n + 1) ]
        else:
            files = [os.path.join(args.path, x) for x in os.listdir(args.path) if 'par' in x]
        states.append(None)
        
    if args.n > 0:
        nindep = args.n
    else:
        nindep = len(files[0]) if multiple_states else len(files)
        
    if multiple_states:
        results = {}
        errors = {}
        for s in states:
            results[s] = [{k:[[] for i in range(args.nbox)] for k in targets} for x in range(len(solvers))]
            errors[s] = [{k:[[] for i in range(args.nbox)] for k in targets} for x in range(len(solvers))]
    else:
        results = [{k:[[] for i in range(args.nbox)] for k in targets} for x in range(len(solvers))]
        errors = [{k:[[] for i in range(args.nbox)] for k in targets} for x in range(len(solvers))]
    
    
    
    for i in range(nindep):
        f = [subdir[i] for subdir in files] if multiple_states else files[i]
        print("read", f)
        traj = Trajectory(f, args.nbox, args.interval)
        for i in range(len(solvers)):
            solver = solvers[i](traj)
            for t in targets:
                for ibox in range(1, args.nbox + 1):
                    ymean, ystd, mad = solver.solve(t, ibox, verbose=verbose)
                    if multiple_states:
                        for j, s in enumerate(states):
                            results[s][i][t][ibox - 1].append(ymean[j])
                            errors[s][i][t][ibox - 1].append(mad[j])                       
                    else:
                        results[i][t][ibox - 1].append(ymean)
                        errors[i][t][ibox - 1].append(mad)
    for s in states:
        if not multiple_states:
            errors_cur = errors
            results_cur = results
        else:
            errors_cur = errors[s]
            results_cur = results[s]
        for t in targets:
            for ibox in range(1, args.nbox + 1):
                print("Partial molar %s for box %d" % (solvers[i].names[t], ibox))
                for i in range(len(solvers)):
                    raw = np.array(errors_cur[i][t][ibox - 1].copy())
                    errors_cur[i][t][ibox - 1] = {}
                    errors_cur[i][t][ibox - 1]['mean'] = np.mean(raw)
                    errors_cur[i][t][ibox - 1]['std'] = np.std(raw)
                    raw = np.array(results_cur[i][t][ibox - 1].copy())
                    results_cur[i][t][ibox - 1] = {}
                    results_cur[i][t][ibox - 1]['mean'] = np.mean(raw, axis=0).ravel().tolist()
                    results_cur[i][t][ibox - 1]['std'] = np.std(raw, axis=0).ravel().tolist()
                    print(solvers[i].name + "".join(["\t%f +/- %f" % (results_cur[i][t][ibox - 1]['mean'][j], 
                                                    results_cur[i][t][ibox - 1]['std'][j]) for j in range(raw.shape[1])])\
                           + "\tError: %f +/- %f" % (errors_cur[i][t][ibox - 1]['mean'], errors_cur[i][t][ibox - 1]['std']))
    
    
    with open(args.path+"_results.json", 'w') as f:
        json.dump({'results': results, 'errors': errors}, f)


    