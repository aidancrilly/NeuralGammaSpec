import numpy as np
from csaps import csaps
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from forward_model import detectormodel

def critial_energy_spectrum(E,Ec,epsilon=1E-11):
    # See equation 5 in paper
    part1 = (1-(0.18**2)*(E/Ec)**2)
    part2 = E**(-0.94)
    output = part1*part2
    output[output < epsilon] = epsilon
    return output

def generate_random_array(mag,length):
    # Random shifts by up and down by mag
    return mag*np.random.choice([-1, 0, 1], size=length)

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def gaussian_decomposition(E, amps, mus, sigma):
    # N.B. single sigma for all gaussians
    total = np.sum(np.array([a * gaussian(E, m, sigma) for a, m in zip(amps, mus)]), axis=0)
    return total

def zeroed_gaussian_amp_guess(n_gaussians):
    # Guess for the Gaussian amplitudes, zeroing out the first 12 and last 20
    # Rather specific to the problem at hand
    p0 = np.ones(n_gaussians)
    p0[:12] = 0.0
    p0[-20:] = 0.0
    return p0

def fit_gaussian_decomposition(E, S, guess_func, n_gaussians=41, mu_range=(25,1025), sigma=25.0):
    mus = np.linspace(mu_range[0], mu_range[1], n_gaussians)
    def _wrap_gaussian_decomposition(E, *amps):
        return gaussian_decomposition(E, amps, mus, sigma)
    p0 = guess_func(n_gaussians)
    popt, pcov = curve_fit(_wrap_gaussian_decomposition, E, S, p0=p0)
    return popt, _wrap_gaussian_decomposition

def generate_noised_spectra_and_signals(E, spectrum_function, epsilon, detector, smoothcontroller = [1E3,1E4,1E5,1E6,1E7,1E8]):

    ground = spectrum_function(E)
    normground = ground/np.mean(ground)
    lognormground = np.log(normground)
    RandomDeltaArray = generate_random_array(epsilon,len(ground))

    mask = (lognormground < -20) & (RandomDeltaArray == -1)
    RandomDeltaArray[mask] = 0

    Variation = np.log(normground) + RandomDeltaArray

    def smoother(x,y,xs,smooth_val):
        SmoothVariation = csaps(x, y, xs, smooth=0.001/smooth_val)
        SmoothVariation = np.exp(SmoothVariation)
        SmoothVariation = SmoothVariation/np.mean(SmoothVariation)
        SmoothVariation = np.log(SmoothVariation)
        return SmoothVariation
    
    listofvariations = [smoother(E,Variation,E,s) for s in smoothcontroller]

    def smooth_signal(SV):
        S = detector.numpy_matmul(np.exp(SV))
        return S/np.mean(S)
        
    listofSs = [smooth_signal(SV) for SV in listofvariations]

    return [listofvariations, listofSs]

def load_traintest_data(stage,seed = 12436,test_size=0.3):
    if stage == 1:
        # Use defaults
        X = np.loadtxt('./InputData/stage1X.csv', delimiter=',')
        Y = np.loadtxt('./InputData/stage1Y.csv', delimiter=',')
        # Additional shuffling
        rng = np.random.default_rng(seed=seed)
        X = rng.shuffle(X)
        Y = rng.shuffle(Y)
    elif stage == 2:
        # Use defaults
        X1 = np.loadtxt('./InputData/stage1X.csv', delimiter=',')
        Y1 = np.loadtxt('./InputData/stage1Y.csv', delimiter=',')
        X2 = np.loadtxt('./InputData/stage2X.csv', delimiter=',')
        Y2 = np.loadtxt('./InputData/stage2Y.csv', delimiter=',')
        X = np.concatenate([X1,X2])
        Y = np.concatenate([Y1,Y2])
        # Additional shuffling
        rng = np.random.default_rng(seed=seed)
        X = rng.shuffle(X)
        Y = rng.shuffle(Y)
    else:
        raise NotImplementedError()
    
    # Do we need this with validation split?
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size)
    return X_train, X_test, y_train, y_test
    
def generate_training_data():
    # Generate training data for the model

    # Fixed gamma energy grid
    Eys = np.arange(0,1000, 1) + 0.5

    # Stage 1 data generation
    # This only includes the radiation reaction model spectrum
    Ecs = 630 - np.geomspace(600, 30, num=30)
    epsilons = range(1,40)

    numrepeats = 10 # repeats at same Ec and Epsilon
    stage1X = []
    stage1Y = []

    for epsilon in epsilons:
        for Ec in Ecs:
            for trial in range(numrepeats):
                spec_model = lambda E: critial_energy_spectrum(E, Ec)
                trialvary, trialSs = generate_noised_spectra_and_signals(Eys,spec_model,epsilon,detectormodel)
                stage1X = stage1X + trialvary
                stage1Y = stage1Y + trialSs

    stage1X = np.array(stage1X)
    stage1Y = np.array(stage1Y)

    np.savetxt('./InputData/stage1X.csv', stage1X, delimiter=',')
    np.savetxt('./InputData/stage1Y.csv', stage1Y, delimiter=',')

    # Stage 2 data generation
    # We add into the radiation reaction model spectrum, data from fitted electron spectra
    Ee_data = np.loadtxt('./InputData/Ee_data.csv', delimiter=',')
    Ees, Ee_spec = Ee_data[:,0], Ee_data[:,1:]
    Nspec = Ee_spec.shape[1]
    # Extend down to zero
    dEe = Ees[1] - Ees[0]
    NE_pad = int(Ees[0]/dEe)
    Ees = np.concatenate((np.arange(0,NE_pad)*dEe, Ees))
    Ee_spec = np.concatenate((np.zeros((NE_pad,Nspec)), Ee_spec), axis=0)
    # High energy masking
    Ee_spec[Ees > 1020.0,:] = 0.0
    # Normalise data
    Ee_spec = Ee_spec / np.mean(Ee_spec, axis=0)

    # Fit the electron spectra
    popts = []
    for i in range(Nspec):
        popt_data, fit_func = fit_gaussian_decomposition(Ees, Ee_spec[:,i], zeroed_gaussian_amp_guess)
        fit_spec = fit_func(Ees, *popt_data)
        normfit_spec = fit_spec / np.mean(fit_spec)
        # Perform some data augmentation
        peak_mask = (Ees > 400) & (Ees < 500)
        meanpeak = np.mean(Ee_spec[peak_mask,i])
        maxpeak  = np.mean(normfit_spec[peak_mask])

        popt.append(popt_data)

        # Using meanpeak
        # Addition of flat low energy peak
        _aug_lowenergypeak = meanpeak - fit_spec
        _aug_lowenergypeak[Ees > 350.0] = 0.0
        _aug_lowenergypeak = normfit_spec + _aug_lowenergypeak
        popt_aug, _ = fit_gaussian_decomposition(Ees, _aug_lowenergypeak, zeroed_gaussian_amp_guess)
        popt.append(popt_aug)

        # Addition of low energy slope
        _aug_lowenergyslope = meanpeak * np.minimum(Ees / 100.0, 1.0) - fit_spec
        _aug_lowenergyslope[Ees > 380.0] = 0.0
        _aug_lowenergyslope = normfit_spec + _aug_lowenergyslope
        popt_aug, _ = fit_gaussian_decomposition(Ees, _aug_lowenergypeak, zeroed_gaussian_amp_guess)
        popt.append(popt_aug)

        # Using maxpeak
        # Addition of flat low energy peak
        _aug_lowenergypeak = maxpeak - fit_spec
        _aug_lowenergypeak[Ees > 350.0] = 0.0
        _aug_lowenergypeak = normfit_spec + _aug_lowenergypeak
        popt_aug, _ = fit_gaussian_decomposition(Ees, _aug_lowenergypeak, zeroed_gaussian_amp_guess)
        popt.append(popt_aug)

        # Addition of low energy slope
        _aug_lowenergyslope = maxpeak * np.minimum(Ees / 100.0, 1.0) - fit_spec
        _aug_lowenergyslope[Ees > 380.0] = 0.0
        _aug_lowenergyslope = normfit_spec + _aug_lowenergyslope
        popt_aug, _ = fit_gaussian_decomposition(Ees, _aug_lowenergypeak, zeroed_gaussian_amp_guess)
        popt.append(popt_aug)
        

    stage2X = []
    stage2Y = []

    epsilons = np.geomspace(1,800, num=40)
    for epsilon in epsilons:
        for popt in popts:
            for trial in range(numrepeats):
                spec_model = lambda E: fit_func(E, *popt)
                trialvary, trialSs = generate_noised_spectra_and_signals(Eys,spec_model,epsilon,detectormodel)
                stage2X = stage2X + trialvary
                stage2Y = stage2Y + trialSs

    stage2X = np.array(stage2X)
    stage2Y = np.array(stage2Y)

    np.savetxt('./InputData/stage2X.csv', stage2X, delimiter=',')
    np.savetxt('./InputData/stage2Y.csv', stage2Y, delimiter=',')

if __name__ == "__main__":
    generate_training_data()