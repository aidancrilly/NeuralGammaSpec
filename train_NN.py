from .data_creation import *
from .forward_model import *
from .loss import *
from .NN_model import *

# Training involves 2 stages:
# 1. Training of a single model with a simplified loss with a subset of data
# 2. Training of an ensemble of models with the full loss and all data

#%%
# Stage 1: Initial Model Training

stage_1_loss = create_custom_loss(detectormodel,loss_weights=[1,0])
X_train, X_test, y_train, y_test = load_traintest_data(stage=1)

InitialModel = NNConvModel(stage_1_loss)
InitialModel.fit(X_train, y_train, nepochs=10, verbose=True, validation_split=0.1)
InitialModel.save_model('./models/NNConvModel_stage1.h5')

# Create a checkpoint for the next stage
checkpoint = './models/startofbeautifulmodel.ckpt'

#%%
# Stage 2: Ensemble Model Training
# This stage will use the initial model as a starting point and train multiple models with the full loss function.

ensemble_size = 100

stage_2_loss = create_custom_loss(detectormodel,loss_weights=[10,70])
X_train, X_test, y_train, y_test = load_traintest_data(stage=2)

checkpoint = 'startofbeautifulmodel.ckpt'

# Model training
for i in range(ensemble_size):
    _NNmodel = NNConvModel(stage_2_loss,weights_file=checkpoint)
    # Relying on keras validation split to handle the randomised split
    _NNmodel.fit(X_train,y_train)
    model_file = f'./models/NNConvModel_{i+1}.h5'
    _NNmodel.save_model(model_file)
