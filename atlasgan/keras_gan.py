from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.backend import expand_dims
from keras.optimizers import Adam
from tensorflow import rank

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self,data):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64
        self.num_glayers = 4
        self.data = data

        optimizer1 = Adam(0.0002, 0.5)
#         optimizer2 = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        print("BUILDING DISCRIMINATOR======>")
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer1,
            metrics=['accuracy'])

        # Build the generator
        print("BUILDING GENERATOR======>")
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(1, 64))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer1)


    def build_generator(self):

# #         model = Sequential()
        
# # #         model.add(Conv2D(256, (5,5), input_shape = [self.img_rows, self.img_cols, 1], activation = 'relu'))
# # #         model.add(MaxPooling2D(pool_size = (2,2)))
# #         model.add(Dense(256, input_dim=self.latent_dim))
# #         model.add(LeakyReLU(alpha=0.2))
# #         model.add(BatchNormalization(momentum=0.8))
# #         model.add(Dense(512))
# #         model.add(LeakyReLU(alpha=0.2))
# #         model.add(BatchNormalization(momentum=0.8))
# #         model.add(Dense(1024))
# #         model.add(LeakyReLU(alpha=0.2))
# #         model.add(BatchNormalization(momentum=0.8))
# #         model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
# #         model.add(Reshape(self.img_shape))
        
# #         model.summary()

# #         noise = Input(shape=(self.latent_dim,))
# # #         print(noise.shape)
# #         img = model(noise)

# #         return Model(noise, img)

#         model = Sequential()

# #         filters = 512
        
# #         model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
# #         model.add(Reshape((8, 8, 128)))
# #         model.add(UpSampling2D())
        
# #         for h in range(self.num_glayers):
# #             model.add(Conv2D(int(filters), kernel_size=5, padding='same'))
# #             model.add(BatchNormalization(momentum=0.8))
# #             model.add(Activation('relu'))
# #             filters /= 2
            
# #         model.add(Activation("relu"))
        
#         model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
#         model.add(Reshape((8, 8, 128)))
#         model.add(UpSampling2D())
#         model.add(Conv2D(128, kernel_size=5, padding="same", input_shape = [64,64,1]))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Activation("relu"))
#         model.add(UpSampling2D())
#         model.add(Conv2D(64, kernel_size=5, padding="same"))
#         model.add(UpSampling2D())
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Conv2D(64, kernel_size=5, padding="same"))
#         model.add(UpSampling2D())
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Conv2D(64, kernel_size=5, padding="same"))
#         model.add(UpSampling2D())
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Conv2D(64, kernel_size=5, padding="same"))
#         model.add(UpSampling2D())
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Activation("relu"))
#         model.add(Conv2D(64, kernel_size=5, padding="same"))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Activation("relu"))
#         model.add(Conv2D(64, kernel_size=3, padding="same"))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Conv2D(64, kernel_size=3, padding="same"))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Activation("relu"))
#         model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
#         model.add(Activation("relu"))

#         model.summary()

#         noise = Input(shape=(self.latent_dim,))
#         print(rank(noise))
#         img = model(noise)

#         return Model(noise, img)

#         model = Sequential()
    
# #         model.add(Conv2D(256, (5,5), input_shape = [self.img_rows, self.img_cols, 1], activation = 'relu'))
#         model.add(Dense(256, input_dim=self.latent_dim))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(512))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(1024))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(np.prod(self.img_shape), activation='tanh'))
#         model.add(Reshape(self.img_shape))

#         model.summary()

#         noise = Input(shape=(self.latent_dim,))
#         img = model(noise)

#         return Model(noise, img)
#         noise_vect_len = 64
        
#         gmodel = Sequential()

#         gmodel.add(Dense(64*64, input_shape=(self.img_rows, self.img_cols, 1)))
#         gmodel.add(BatchNormalization())
#         gmodel.add(Activation('relu'))
# #         gmodel.summary()
#         gmodel.add(Reshape((64,64,64*64)))
#         gmodel.add(Conv2DTranspose(128,5,strides=2, padding='same'))
#         gmodel.add(BatchNormalization())
#         gmodel.add(Activation('relu'))
#         gmodel.add(Conv2DTranspose(64,5,strides=2, padding='same'))
#         gmodel.add(BatchNormalization())
#         gmodel.add(Activation('relu'))
#         gmodel.add(Conv2DTranspose(32,5,strides=2, padding='same'))
#         gmodel.add(BatchNormalization())
# #         gmodel.add(Activation('relu'))
# #         gmodel.add(Conv2DTranspose(1,5,strides=2, padding='same'))
#         gmodel.add(Activation('tanh'))

#         gmodel.summary()

#         noise = Input(shape=(self.img_rows, self.img_cols, 1))
#         return Model(noise, gmodel(noise))

#         noise_vect_len = 64
        
#         gmodel = Sequential()

#         gmodel.add(Dense(4*4*256, input_shape=(1,noise_vect_len)))
#         gmodel.add(BatchNormalization())
# #         gmodel.add(Activation('relu'))
#         gmodel.add(Activation('sigmoid'))
#         gmodel.add(Reshape((4,4,256)))
#         gmodel.add(Conv2DTranspose(128,5,strides=2, padding='same'))
#         gmodel.add(BatchNormalization())
# #         gmodel.add(Activation('relu'))
#         gmodel.add(Activation('tanh'))
#         gmodel.add(Conv2DTranspose(64,5,strides=2, padding='same'))
#         gmodel.add(BatchNormalization())
# #         gmodel.add(Activation('relu'))
#         gmodel.add(Activation('relu'))
#         gmodel.add(Conv2DTranspose(32,5,strides=2, padding='same'))
#         gmodel.add(BatchNormalization())
# #         gmodel.add(Activation('relu'))
#         gmodel.add(Activation('sigmoid'))
#         gmodel.add(Conv2DTranspose(1,5,strides=2, padding='same'))
#         gmodel.add(Activation('sigmoid'))

#         gmodel.summary()

#         noise = Input(shape=(1,noise_vect_len))
#         return Model(noise, gmodel(noise))
        
        model = Sequential()

        model.add(Dense(1024 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 1024)))
        model.add(UpSampling2D())
        model.add(Conv2D(1024, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)


    def build_discriminator(self):

#         model = Sequential()

#         model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
#         model.add(ZeroPadding2D(padding=((0,1),(0,1))))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
#         model.add(ZeroPadding2D(padding=((0,1),(0,1))))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
#         model.add(ZeroPadding2D(padding=((0,1),(0,1))))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
# #         model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
# #         model.add(BatchNormalization(momentum=0.8))
# #         model.add(LeakyReLU(alpha=0.2))
# #         model.add(Dropout(0.25))
#         model.add(Flatten())
#         model.add(Dense(1, activation='sigmoid'))

#         model.summary()

#         img = Input(shape=self.img_shape)
#         validity = model(img)

#         return Model(img, validity)

#         model = Sequential()

#         model.add(Flatten(input_shape=self.img_shape))
#         model.add(Dense(2048))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(1024))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(512))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(256))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(1, activation='sigmoid'))
#         model.summary()

#         img = Input(shape=self.img_shape)
#         validity = model(img)

#         return Model(img, validity)
#         dmodel = Sequential()
        
#         dmodel.add(Dense(64*64, input_shape = (self.img_rows, self.img_cols, 1)))
#         dmodel.add(BatchNormalization())
#         dmodel.add(Activation('relu'))
#         dmodel.add(Reshape((8,8,64*64*64)))
#         dmodel.add(Conv2D(filters=32, kernel_size=5, strides=2, padding='same', input_shape=(64,64,1)))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=256, kernel_size=5, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Flatten())
#         dmodel.add(Dense(1))
#         dmodel.add(Activation('sigmoid'))

#         dmodel.summary()

#         img = Input(shape=(self.img_rows,self.img_cols, 1))

#         return Model(img, dmodel(img))

#         dmodel = Sequential()

#         dmodel.add(Conv2D(filters=32, kernel_size=5, strides=2, padding='same', input_shape=(64,64,1)))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=128, kernel_size=4, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=256, kernel_size=5, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Conv2D(filters=512, kernel_size=5, strides=2, padding='same'))
#         dmodel.add(BatchNormalization())
#         dmodel.add(LeakyReLU(alpha=0.2))
#         dmodel.add(Flatten())
#         dmodel.add(Dense(1))
#         dmodel.add(Activation('sigmoid'))

#         dmodel.summary()

#         img = Input(shape=(64,64,1))

        model = Sequential()

        model.add(Conv2D(94, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(188, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(376, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(752, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(1504, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    
    #Wahid's draw_image function
    def draw_image(x, vmin=None, vmax=None, xlabel='$\eta$', ylabel='$\phi$',
                   draw_cbar=True, ax=None, figsize=(5,4), **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(x.T, extent=[-2.5, 2.5, -3.15, 3.15],
                        norm=LogNorm(vmin, vmax), aspect='auto',
                        origin='lower', **kwargs)
        if draw_cbar:
            cbar = plt.gcf().colorbar(cax, ax=ax)
            cbar.set_label('Energy')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    def train(self, steps, batch_size=128): 
              #sample_interval=50):

        # Load the dataset
        X_train = self.data
        
        # Rescale -1 to 1
#         X_train = np.multiply(X_train, 1. / (127.5 - 1.))
        X_train = np.expand_dims(X_train, axis=3)
        print("Training data shape: ", type(X_train))
        print("Training data shape: ", X_train.shape)
        print("Training data min: ", X_train.min())
        print("Training data max: ", X_train.max())
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        

        for step in range(steps):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

#             noise = np.random.normal(0, 1, (batch_size, 1, 64))
            noise = np.random.normal(0,1,(128,64))
#             print(noise.shape)
#             noise = expand_dims(noise, 0)

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)
            
            #TODO: Add label flipping here:
            
            
#             Train the discriminator
            flip = np.random.randint(0, 100)
            if flip <= 5:
                d_loss_real = self.discriminator.train_on_batch(imgs, fake)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, valid)
               
            else:
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
         
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 1, 64))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if ( (step%1) == 0 ): print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (step, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
#             if epoch % sample_interval == 0:
#                 self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(np.expand_dims(noise,1))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
#         fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)