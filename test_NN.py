from .data_creation import *
from .forward_model import *
from .loss import *
from .NN_model import *

from glob import glob
import matplotlib.pyplot as plt

loss = create_custom_loss(1,detectormodel,loss_weights=[10,70])

model_files = glob('./models/NNConvModel_*.h5')

EnsembleModel = Ensemble_NNConvModel(loss,model_files)

X_train, X_test, y_train, y_test= load_traintest_data(0)

y_pred = EnsembleModel(X_test)

plt.plot(X_test,y_pred)
plt.plot(X_test,y_test)

plt.show()