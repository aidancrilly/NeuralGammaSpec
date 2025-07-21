from .data_creation import *
from .forward_model import *
from .loss import *
from .NN_model import *

ensemble_size = 100

loss = create_custom_loss(1,detectormodel,loss_weights=[10,70])

checkpoint = 'startofbeautifulmodel.ckpt'

model_files = []

# Model training
for i in range(ensemble_size):
    _NNmodel = NNConvModel(loss,weights_file=checkpoint)
    X_train, X_test, y_train, y_test = load_traintest_data(i)
    _NNmodel.fit(X_train,y_train)
    model_file = f'./models/NNConvModel_{i+1}.h5'
    _NNmodel.save_model(model_file)
