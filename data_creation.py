import numpy as np
from csaps import csaps
from sklearn.model_selection import train_test_split

def critial_energy_spectrum(E,Ec,epsilon=1E-11):
    # See equation 5 in paper
    part1 = (1-(0.18**2)*(E/Ec)**2)
    part2 = E**(-0.94)
    output = part1*part2
    output[output < epsilon] = epsilon
    return output

def generate_random_array(mag,length):
    return mag*np.random.choice([-1, 0, 1], size=length)

def generate_random_fitted_spec_and_signals(E, epsilon, Ec, detector, smoothcontroller = [1E3,1E4,1E5,1E6,1E7,1E8]):

    ground = critial_energy_spectrum(E, Ec)
    normground = ground/np.mean(ground)
    lognormground = np.log(normground)
    RandomDeltaArray = generate_random_array(epsilon,len(ground))
    # ...
    for i in range(len(RandomDeltaArray)):
        if i>600 and lognormground[i]<-20 and RandomDeltaArray[i] == -1:
            RandomDeltaArray[i]=0

    Variation = np.log(normground) + RandomDeltaArray

    def smoother(x,y,xs,smooth_val):
        SmoothVariation = csaps(x, y, xs, smooth=0.001/smooth_val)
        SmoothVariation = np.exp(SmoothVariation)
        SmoothVariation = SmoothVariation/np.mean(SmoothVariation)
        SmoothVariation = np.log(SmoothVariation)
        return SmoothVariation
    
    listofvariations = [smoother(E,Variation,E,s) for s in smoothcontroller]

    def smooth_signal(SV):
        S = detector(np.exp(SV))
        return S/np.mean(S)
        
    listofSs = [smooth_signal(SV) for SV in listofvariations]

    return [listofvariations, listofSs]

def load_traintest_data(index,test_size=0.3,dataset=None):
    if dataset is None:
        # Use defaults
        X = np.loadtxt('./InputData/bunchofspectradata1.csv', delimiter=',')
        Y = np.loadtxt('./InputData/bunchofsignalsdata1.csv', delimiter=',')
        # Additional shuffling
        rng = np.random.default_rng(seed=(index+3)*420)
        X = rng.shuffle(X)
        Y = rng.shuffle(Y)
    else:
        raise NotImplementedError()
    
    # Do we need this with validation split?
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size)
    return X_train, X_test, y_train, y_test
    