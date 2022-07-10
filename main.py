#!/usr/bin/env python
# coding: utf-8

# Additions and alterations done by: Fay Beening
# Original author: Margaret Maynard-Reid ([@margaretmz](https://twitter.com/margaretmz))
#
# This Colab notebook is a **Wasserstein GAN with Gradient Penality (WGAN-GP)** implementation with TensorFlow 2/ Keras, trained to generate 64x64 anime faces. It is associated with the [Anime Faces with WGAN and WGAN-GP](TODO) blog post published on 2022-02-07, as part of the [PyImageSearch University](https://www.pyimagesearch.com/pyimagesearch-university/) GAN series.


import tensorflow as tf
import os
import glob
import shutil
import serial

from tensorflow import keras
from functions.generator import build_generator
from functions.discriminator import build_critic
from functions.DCGAN import WGAN_GP
from functions.GANmonitor import GANMonitor
from functions.imageRenew import image_renew
from functions.imageRenew import image_select

print(tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

folder_files = os.listdir("./pr_wikiart/")  # You can also use full path
dst_dir = "./working_imgs"
data_dir = "./working_imgs_copy/"

LATENT_DIM = 128
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) # weight initializer for G per DCGAN paper
CHANNELS = 3 # number of channels, 1 for gray scale and 3 for color images
LR = 0.0002  # WGAN-GP paper recommends lr of 0.0002
NUM_ITER = 4000 #number of iterations
NUM_EPOCHS = 10  # number of epochs


# generator = tf.keras.models.load_model('generator.h5')
generator = build_generator()
generator.summary()

critic = build_critic(128, 128, 3)
critic.summary()

wgan_gp = WGAN_GP(critic=critic,
                  generator=generator,
                  latent_dim=LATENT_DIM,
                  critic_extra_steps=5)


# Wasserstein loss for the critic
def d_wasserstein_loss(pred_real, pred_fake):
    real_loss = tf.reduce_mean(pred_real)
    fake_loss = tf.reduce_mean(pred_fake)
    return fake_loss - real_loss


# Wasserstein loss for the generator
def g_wasserstein_loss(pred_fake):
    alt_g_loss = tf.reduce_mean(pred_fake)
    return -alt_g_loss

d_optimizer = keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)
g_optimizer = keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)

wgan_gp.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, d_loss_fn=d_wasserstein_loss, g_loss_fn=g_wasserstein_loss)

# innnitial seletcion of images. random.
train_images = image_select()

# make iteration files
os.mkdir("./epochFiles")
for t in range(0, NUM_ITER):
    os.mkdir("./epochFiles/iteration_{}".format(t))

# delete files within iteration files
# for filez in range(5000):
#     for filezz in range(1):
#         os.remove("./epochFiles/iteration_{}/epoch_00{}.png".format(filez, filezz))

for x in range(NUM_ITER):
    train_images = image_renew()

    history = wgan_gp.fit(train_images, epochs=NUM_EPOCHS, callbacks=[GANMonitor(num_img=9, latent_dim=LATENT_DIM)])

    movefilepath = './epochFiles/iteration_{}'.format(x)
    for file in glob.glob("./epoch_000.png"):
        shutil.copy(file, movefilepath)
    print('\n ////////////////////////////////////////// ITERATION {} /////////////////////////////////////////// \n'.format(x))





