import os
import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import logging
import numpy as np
import scipy.io as sio

# randomly choose a mask from 4 masks
def generate_mask_path(num_masks):
    a=np.random.randint(num_masks)+1
    mask_path = ("Data/mask{:d}.mat".format(a))
    return mask_path

def generate_masks(mask_path, batch_size, energy=False): #function to generate masks
    
    # load the mask.mat to mask
    mask = sio.loadmat(mask_path) 
    mask = mask['mask']
    
    # Add more channels 
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,28)) 
    # exchange 2,0,1 dimensions, mask3d (28,256,256) now
    mask3d = np.transpose(mask3d, [2, 0, 1]) 
    # Change mask3d into tf Tensor
    mask3d = tf.convert_to_tensor(mask3d, dtype='float32') 
    # Get [number of channels=28, height, width]
    [nC, H, W] = mask3d.shape 
    
    # Expand to mask3d_batch
    mask3d_batch = tf.tile(mask3d[np.newaxis,...],[batch_size,1,1,1])
    # if using energy normalization, return extra mask_s
    if energy:
        temp = shift_energy(mask3d, 2)
        mask_s = np.sum(temp, axis=0)
    else:
        mask_s = None
   
    return mask3d_batch, mask_s

def LoadTraining(path, short=False):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
        if short and i == 30:
            break

    return imgs

def LoadTest(path_test, patch_size):
    #Find the path of the test
    scene_list = os.listdir(path_test)
    #sort the list
    scene_list.sort()
    #Define test data
    test_data = np.zeros((len(scene_list), patch_size, patch_size, 28))
    for i in range(len(scene_list)):
        #Find the path of scene
        scene_path = path_test + scene_list[i]
        #Load the imgs
        img_dict = np.load(scene_path, allow_pickle=True).item()
        # img as a dictionary
        img = img_dict['img']
        #img = img/img.max()
        test_data[i,:,:,:] = img
        #Print the order number, img shapes, maximum of img, minimum of img
        print(i, img.shape, img.max(), img.min())
        
    #Get test_data in tf format
    test_data = tf.convert_to_tensor(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data




#Functions to get psnr
def psnr(img1, img2):
    # Define the psnr_list
    psnr_list = []
    # iterate all img1
    for i in range(img1.shape[0]):
        total_psnr = 0
        PIXEL_MAX = img2[i,:,:,:].max()
        # iterate 28 channels
        for ch in range(28):
            # formula to calculate psnr
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            # formula to calculate psnr
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        # contain psnr in the psnr_list
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

#Find psnr for a single image
def tf_psnr(img, ref):
    img, ref = img.numpy(), ref.numpy()
    #Get the number of channels
    nC = img.shape[0]
    #maximum pixel
    pixel_max = np.max(ref)
    #define variable psnr
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr/nC




def gaussian(window_size, sigma):
    gauss = np.array(([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]))
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = np.expand_dims(gaussian(window_size, 1.5),1)
    _2D_window = np.dot(_1D_window, _1D_window.T)[np.newaxis, np.newaxis, ...]
    window = np.ascontiguousarray(np.tile(_2D_window, [channel,1,1,1]), dtype='float32')
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    # initialize a 2d conv between img and window
    conv = Conv2D(channel, window_size, padding = 'same', data_format = 'channels_first', use_bias=False, groups=channel,
                  input_shape = img1.shape)
    conv(img1)
    conv.set_weights([np.transpose(window, [2,3,1,0])])
    
    mu1 = conv(img1)
    mu2 = conv(img2)

    mu1_sq = np.power(mu1,2)
    mu2_sq = np.power(mu2,2)
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = conv(img1*img1) - mu1_sq
    sigma2_sq = conv(img2*img2) - mu2_sq
    sigma12 = conv(img1*img2) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return np.mean(ssim_map)
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.shape
    window = create_window(window_size, channel)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def tf_ssim(img, ref):
    return ssim(np.expand_dims(img,0), np.expand_dims(ref,0))#Use self-defined ssim function





def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename#Find the time 


def shuffle_crop(train_data, batch_size, patch_size):#shuffle and crop images
    
    index = np.random.choice(np.arange(len(train_data)), batch_size)#index of training data
    processed_data = np.zeros((batch_size, patch_size, patch_size, 28), dtype=np.float32)#define processed data,
    #batch_size*patch_size*patch_size*28,1*256*256*28
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape#height,  width of train_data
        x_index = np.random.randint(0, h - patch_size)#get x_index
        y_index = np.random.randint(0, w - patch_size)#get y_index
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + patch_size, y_index:y_index + patch_size, :]  
        # process train data by shuffling and cropping
    gt_batch = tf.convert_to_tensor(np.transpose(processed_data, (0, 3, 1, 2)))# Get ground truth batch
    return gt_batch


def gen_meas_tf(data_batch, mask3d_batch, mask_s=None, is_training=True):
    nC = data_batch.shape[1]#Get number of channels
    [batch_size, nC, H, W] = data_batch.shape
    if is_training is False:#Without training
        [batch_size, nC, H, W] = data_batch.shape# Get batch_size, number of channels, Heights, and width
        mask3d_batch = np.tile(mask3d_batch[0:1,...],[batch_size,1,1,1])
    
    temp = shift(mask3d_batch*data_batch, 2)#shift mask3d_batch*data_batch
    meas = np.sum(temp, 1)/nC*2
    
    # if mask_s == None, it means do not use energy normalization
    if mask_s is None:
        y_temp = shift_back(meas)
        PhiTy = np.multiply(y_temp, mask3d_batch)
    # if not, then it means using energy normalization, do extra mask computing
    else:
        meas_re = np.divide(meas,mask_s)
        y_temp = shift_back(meas_re)#shift meas
        meas_re = shift_back_energy(meas_re)
        meas_re = meas_re/nC*2
        meas_re = np.reshape(meas_re,[batch_size,1,H,W])
        PhiTy = np.multiply(y_temp, mask3d_batch)#Get input F_Y, multiply measurement y and mask
        PhiTy = np.concatenate([meas_re,PhiTy],axis=1)
    return PhiTy

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape#define shift function
    output = np.zeros([bs, nC, row, col+(nC-1)*step], dtype='float32')
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]#the way of shifting difference between lamd_n and lamda_c
    return output

def shift_energy(inputs, step=2):
    [nC, row, col] = inputs.shape #define shift function
    #nC=28
    output = np.zeros([nC, row, col+(nC-1)*step], dtype='float32')
    for i in range(nC):
        output[:,:,step*i:step*i+col] = inputs[i,:,:]#the way of shifting difference between lamd_n and lamda_c
    return output

def shift_back_energy(inputs,step=2):
    [bs,row, col] = inputs.shape
    nC = 28
    output = np.zeros([bs, nC, row, col-(nC-1)*step], dtype='float32')
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]#function to shift back
    #inputs=shift_back_energy(inputs)
    #pdb.set_trace()
    #inputs=torch.reshape(inputs,[bs,1,256,256])
    output=np.sum(output, axis=1)
    
    return output

def shift_back(inputs,step=2):
    [bs, row, col] = inputs.shape
    nC = 28
    output = np.zeros([bs, nC, row, col-(nC-1)*step], dtype='float32')
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]#function to shift back
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'#log file contains information of model_path
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
