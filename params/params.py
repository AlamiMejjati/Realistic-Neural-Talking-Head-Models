#K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape, path_to_mp4

#number of frames to load
K = 8

#path to main weight
path_to_chkpt = 'model_weights.tar'

#path to backup
path_to_backup = 'backup_model_weights.tar'

#CHANGE first part
path_to_Wi = ""+"Wi_weights"
#path_to_Wi = "test/"+"Wi_weights"


#batch size per gpu
batch_size = 2

#nb workers

nb_workers = 8

#dataset save path
path_to_preprocess = '/home/youssef/Documents/phdYoop/datasets/vox2/dev/ims'

#default for Voxceleb
frame_shape = 224

#path to dataset
path_to_mp4 = '/home/youssef/Documents/phdYoop/datasets/vox2/dev/mp4'

path_to_save = './train_log'

#frequency of printing
print_freq = 100

#frequency of image logs
log_freq = 1000

# gpus to use

use_gpus = '0,1,2,3'