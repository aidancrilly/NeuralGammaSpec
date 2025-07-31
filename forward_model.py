import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

class DetectorModel():

    def __init__(self,Rfilename,Efilename):
        self.R_filename = Rfilename
        self.E_filename = Efilename
        self.R = np.loadtxt(self.R_filename, delimiter=',').T
        self.E = np.loadtxt(self.E_filename, delimiter=',')
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

    def numpy_matmul(self,N):
        S = np.matmul(self.R,N)
        return S

    def __call__(self,N):
        S = tf.matmul(self.R_tensor,N)
        return S

# Default set up
E_detector = np.arange(0, 1000, 1) + 0.5
detectormodel = DetectorModel('./InputData/Rfinal.csv','./InputData/Ey_R.csv')
detectormodel.regrid(E_detector)
detectormodel.tensorize()

