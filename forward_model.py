import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

class DetectorModel():

    def __init__(self,filename,dE=1.0,E0=1.0):
        self.R_filename = filename
        self.R = np.loadtxt(self.R_filename, delimiter=',').T
        self.E = E0 + np.arange(self.R.shape[0])*dE
        self.NE = self.R.shape[1]
        assert self.NE == len(self.E)
        self.ND = self.R.shape[0]
    
    def regrid(self,new_E):
        new_R = np.zeros((self.ND,new_E.shape[0]))
        for i in range(self.ND):
            cs = CubicSpline(self.E,self.R[i,:])
            new_R[i,:] = cs(new_E)
        self.E = new_E
        self.R = new_R

    def tensorize(self):
        self.R_tensor = tf.convert_to_tensor(self.R, tf.float32)

    def __call__(self,N):
        S = tf.matmul(N,self.R_tensor)
        return S
    
# Default set up
E_detector = np.arange(0, 1000, 1) + 0.5
detectormodel = DetectorModel('./InputData/Rfinal.csv')
detectormodel.regrid(E_detector)
detectormodel.tensorize()