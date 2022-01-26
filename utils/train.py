import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import time
import scipy.io as scio


from utils.utils import *
from utils.model import SSI_RES_UNET


class Train_Model():
    def __init__(self, mask_num, energy, mix, train_data, test_data, model_name):
        # data definition
        self.train_data = train_data
        self.test_data = test_data
        self.mask_path = "Data/mask"+str(mask_num)+".mat"   # Mask path, mask num can be [1,2,3,4]
        self.save_path = "models/" + model_name
        
        # training setting
        self.in_c = 28 + int(energy)               # input channels, should be 28 if not energy and 29 if energy
        self.patch_size = 256                      # model input size
        self.max_epoch = 200                       # training epochs
        self.learning_rate = 0.0004                # Learning rate=4*10^-4
        self.epoch_sam_num = 5000                  # how many samples one epoch will do
        self.batch_size = 4                        # Batch size
        self.batch_num = int(np.floor(self.epoch_sam_num/self.batch_size))
#         self.batch_num = 2
        self.energy = energy                       # if use energy normalization
        self.mix = mix
        
        # fine tune params
        self.last_train = 0                        # for finetune
        self.model_save_filename = ''              # for finetune
        self.logger = None                         # logger to record training info
        
        self.optimizer = keras.optimizers.Adam(self.learning_rate)  # use adam optimizer
        self.mse = keras.losses.MeanSquaredError()             # use mse loss
    
    def initialize_model(self):
        module = SSI_RES_UNET()
        # model input should be [28,256,256] by default
        self.model = module.create_model([self.in_c,self.patch_size,self.patch_size])
        print("="*10,"Model Structure","="*10)
        self.model.summary()
    
    def optimize(self, epoch, logger, mask3d_batch, mask_s):
        # define loss and time record
        epoch_loss = 0
        begin = time.time()

        # training over batches
        for i in range(self.batch_num):
            with tf.GradientTape() as tape:
                # get batch ground truth
                gt_batch = shuffle_crop(self.train_data, self.batch_size, self.patch_size)
                gt = tf.Variable(gt_batch, dtype='float32')
                #mask3d_batch = generate_masks(mask_path, batch_size)#generate 3d masks

                # generate corresponding CSCCI compressed representation
                y = gen_meas_tf(gt, mask3d_batch, mask_s, is_training = True)
                # put into the model
                model_out = self.model(y)

                #Calculate the loss 
                loss = tf.sqrt(self.mse(model_out, gt))
                #sum epoch loss
                epoch_loss += loss

                #Backpropagation
                gradients = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        end = time.time()
    #     logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss/batch_num, (end - begin)))
        print("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss/self.batch_num, (end - begin)))
        
    def test(self, epoch, logger, mask3d_batch, mask_s):
        # initialize lists to save results
        psnr_list, ssim_list = [], []

        #Get the spatial shifting masks
        test_gt = self.test_data
        test_PhiTy = gen_meas_tf(test_gt, mask3d_batch, mask_s, is_training = False)

        begin = time.time()
        # Get the output of the model
        model_out = self.model(test_PhiTy)
        end = time.time()

        for k in range(test_gt.shape[0]):
            psnr_val = tf_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])#Output the psnr value
            ssim_val = tf_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])#Output the  ssim value
            psnr_list.append(psnr_val)#Append the psnr value
            ssim_list.append(ssim_val)#Append the ssim value
        pred = np.transpose(model_out.numpy(), (0, 2, 3, 1)).astype(np.float32)
        truth = np.transpose(test_gt.numpy(), (0, 2, 3, 1)).astype(np.float32)
        psnr_mean = np.mean(np.asarray(psnr_list))
        ssim_mean = np.mean(np.asarray(ssim_list))
    #     logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, psnr_mean, ssim_mean, (end - begin)))
        print('===> testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(psnr_mean, ssim_mean, (end - begin)))
        return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)
    
    def train(self):
        # define saving path
        if self.model_save_filename == '':
            date_time = str(datetime.datetime.now())
            date_time = time2file_name(date_time)
        else:
            date_time = self.model_save_filename
        result_path = 'recon' + '/' + date_time
        model_path = 'model' + '/' + date_time
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logger = gen_log(model_path)
    #     logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))

        #
        psnr_max = 0

        # start training
        print("="*10,"Training start","="*10)
        if self.energy:
            print("Energy Normalization Applied")
        if self.mix:
            print("Mix Training Applied")
        for epoch in range(self.last_train + 1, self.last_train + self.max_epoch + 1):
            if self.mix:
                mask_path = generate_mask_path(4)
            else:
                mask_path = self.mask_path
            mask3d_batch, mask_s = generate_masks(mask_path, self.batch_size, self.energy) #generate 3d masks
            self.optimize(epoch, logger, mask3d_batch, mask_s)
            (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = self.test(epoch, logger, mask3d_batch, mask_s)

            if psnr_mean > psnr_max:
                psnr_max = psnr_mean
                if psnr_mean > 27 :
                    name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                    scio.savemat(name, {'truth':truth, 'pred': pred, 'psnr_list':psnr_all, 'ssim_list':ssim_all})
                    checkpoint(epoch, model_path, logger)
            lr_epoch=50
            lr_scale=0.5#Halve by every 50 epochs
            if (epoch % lr_epoch == 0) and (epoch < 200):
                self.learning_rate = self.learning_rate * lr_scale
                logger.info('Current learning rate: {}\n'.format(learning_rate))
        
        # After training finished, save model
        self.save_model()
                
    def save_model(self):
        tf.keras.models.save_model(self.model, self.save_path)
        