
#========================== public configure ==========================
IMG_SIZE = (512, 512)
TOTAL_EPOCH = 600
INITAL_EPOCH_LOSS = 100000
NUM_EARLY_STOP = 60
NUM_UPDATE_LR = 100
BINARY_CLASS = 1
BATCH_SIZE = 2
learning_rates= 1e-3

# ===================  experiment configure && saved path =========================
DATA_SET = 'drive'
visual_samples = '/root/daima/dilation_supervised/log/visual_samples/'
saved_path = '/root/daima/dilation_supervised/log/weights_save/'+ DATA_SET + '/'
visual_results = '/root/daima/dilation_supervised/log/visual_results/'+ DATA_SET + '/'

# experiment configure of stare dataset
resize_stare = 592
resize_size_stare = (resize_stare, resize_stare)
size_h_stare, size_w_stare = 605, 700

# experiment configure of drive dataset
resize_drive = 512
resize_size_drive = (resize_drive, resize_drive)
size_h, size_w = 584, 565

# experiment configure of chasedb dataset
resize_chasedb = 960
resize_size_chasedb = (resize_chasedb, resize_chasedb)
size_h_chasedb, size_w_chasedb = 960, 999

# experiment configure of hrf dataset
resize_hrf = 960
resize_size_hrf = (resize_hrf, resize_hrf)
size_h_hrf, size_w_hrf = 2336, 3504

# saved path of chasedb dataset
path_image_chasedb = './dataset1/npy/CHASEDB/tempt/train_image_save.npy'
path_label_chasedb = './dataset1/npy/CHASEDB/tempt/train_label_save.npy'
path_test_image_chasedb = './dataset1/npy/CHASEDB/tempt/test_image_save.npy'
path_test_label_chasedb = './dataset1/npy/CHASEDB/tempt/test_label_save.npy'
path_val_image_chasedb = './dataset1/npy/CHASEDB/tempt/val_label_save.npy'
path_val_label_chasedb = './dataset1/npy/CHASEDB/tempt/val_label_save.npy'

# saved path of drive dataset
path_image_drive = './dataset1/npy/DRIVE/tempt/train_image_save.npy'
path_label_drive = './dataset1/npy/DRIVE/tempt/train_label_save.npy'
path_test_image_drive = './dataset1/npy/DRIVE/tempt/test_image_save.npy'
path_test_label_drive = './dataset1/npy/DRIVE/tempt/test_label_save.npy'
path_val_image_drive = './dataset1/npy/DRIVE/tempt/val_label_save.npy'
path_val_label_drive = './dataset1/npy/DRIVE/tempt/val_label_save.npy'


# saved path of stare dataset
path_image_stare = './dataset1/npy/STARE/tempt/train_image_save.npy'
path_label_stare = './dataset1/npy/STARE/tempt/train_label_save.npy'
path_test_image_stare = './dataset1/npy/STARE/tempt/test_image_save.npy'
path_test_label_stare = './dataset1/npy/STARE/tempt/test_label_save.npy'
path_val_image_stare = './dataset1/npy/STARE/tempt/val_label_save.npy'
path_val_label_stare = './dataset1/npy/STARE/tempt/val_label_save.npy'

# saved path of hrf dataset
path_image_hrf = './dataset1/npy/HRF/tempt/train_image_save.npy'
path_label_hrf = './dataset1/npy/HRF/tempt/train_label_save.npy'
path_test_image_hrf = './dataset1/npy/HRF/tempt/test_image_save.npy'
path_test_label_hrf = './dataset1/npy/HRF/tempt/test_label_save.npy'
path_val_image_hrf = './dataset1/npy/HRF/tempt/val_label_save.npy'
path_val_label_hrf = './dataset1/npy/HRF/tempt/val_label_save.npy'

total_drive = 20
total_chasedb = 8
total_stare = 1
total_hrf = 15

Classes_drive_color = 5
