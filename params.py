"""Params for ADDA."""

#params for the size of different data loader 暂时用不到
# n_src_trn_samples = 314
# n_src_eval_samples = 78
# n_tgt_trn_samples = 160
# n_src_eval_samples = 40

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# params for source dataset
src_dataset = "MNIST"
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots_0419_3"
d_input_dims =  50  #40
d_hidden_dims = 40
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 500      #####1
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 2000         #####2
log_step = 2   #100
save_step = 100
manual_seed = 2020   #None

# params for optimizing models
d_learning_rate = 1e-4    #####4
c_learning_rate = 1e-4   #####4
beta1 = 0.9
beta2 = 0.99
