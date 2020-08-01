
from __future__ import print_function, division

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, UpSampling1D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.datasets import mnist

import matplotlib.pyplot as plt

import sys

from os import listdir
from os import remove

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (100,)

        model = Sequential()
        model.add(Dense(4*4*512, input_shape=noise_shape,  use_bias=False))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Reshape((4,4,512)))

        model.add(Conv2DTranspose(256,5, strides=2, padding= "same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.01))
        assert model.output_shape == (None, 8, 8,256)

        model.add(Conv2DTranspose(128, 5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.01))
        assert model.output_shape == (None, 16, 16, 128)

        model.add(Conv2DTranspose(64, 5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.01))
        assert model.output_shape == (None, 32, 32, 64)

        model.add(Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"))
        assert model.output_shape == (None,64, 64, 1)

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(16, 3, activation="relu", input_shape=self.img_shape))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(16, 3, activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, 3, activation="relu"))
        #model.add(Conv2D(64, 3, activation="relu"))
        #model.add(Conv2D(64, 3, activation="relu"))
        model.add(MaxPooling2D(4, 4))
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)



    def train(self, epochs, batch_size, save_interval):

        #List of available images
        dir = r"C:\Users\jdeek\PycharmProjects\Datascience\sudoArt\Redditors Art"
        real_images = listdir(dir)
        loaded_images = list()

        for image_name in real_images:
            print(image_name)
            try:
                temp = load_img(dir+"\\"+str(image_name), color_mode = "grayscale", target_size = (self.img_rows, self.img_cols))
                loaded_images.append(img_to_array(temp))
            except:
                print("invalid_file")

        # Load the dataset

        X_train= np.array(loaded_images[2:])

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            # Generate a half batch of new images

            gen_imgs = self.generator.predict(noise)

            #Soften Labels

            trues = np.ones(half_batch)#- (abs(np.random.normal(0, .1, size=half_batch)))
            falses = np.zeros(half_batch)# + (abs(np.random.normal(0, .1, size=half_batch)))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, trues)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,falses)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.combined.train_on_batch(noise, valid_y)


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def testG(self):
        noise = np.random.normal(0, 1,(1, 100))
        image = self.generator.predict(noise)
        plt.imshow(image[0][:,:,0], interpolation="nearest", cmap='gray')


    def save_imgs(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(r"C:\Users\jdeek\OneDrive\Desktop\fakes\aug"[:-1]+ str(epoch))
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=60000, batch_size=1024, save_interval=50)

