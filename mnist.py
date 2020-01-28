# Tell Google Colaboratoy what version we want to use
%tensorflow_version 1.x

# First we need to import a few libraries which
# help us to do the actual math and machine learning parts
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# --------------------------------------------------

# Load the dataset:
# MNIST database of handwritten digits. 60,000 28x28 grayscale images of the
# 10 digits, along with a test set of 10,000 images.
(X_train_raw, _), (_, _) = mnist.load_data()

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

# Lets see what is inside!
X_train_raw[0]

# --------------------------------------------------

# This part is just for fun!
# We replace all numbers from 1-255 with 1 and print them out.
# With this trick we can actually make the image "visible" as text.

# Pick an index out of the 60000 entries
dataset_index = 512

# Print what is inside ..
for x in range(0, img_cols):
  for y in range(0, img_rows):
    char = '0' if X_train_raw[dataset_index][x][y] == 0 else '1'
    print(char, end = '')
  print() # .. newline
  
# --------------------------------------------------

latent_dim = 100
optimizer = Adam(0.0002, 0.5)

# --------------------------------------------------

# Build and compile the discriminator
dis_model = Sequential()

dis_model.add(Flatten(input_shape=img_shape))
dis_model.add(Dense(512))
dis_model.add(LeakyReLU(alpha=0.2))
dis_model.add(Dense(256))
dis_model.add(LeakyReLU(alpha=0.2))
dis_model.add(Dense(1, activation='sigmoid'))

dis_model.summary()

dis_img = Input(shape=img_shape)
validity = dis_model(dis_img)

discriminator = Model(dis_img, validity)

discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
                      
# --------------------------------------------------

# Build the generator
gen_model = Sequential()

gen_model.add(Dense(256, input_dim=latent_dim))
gen_model.add(LeakyReLU(alpha=0.2))
gen_model.add(BatchNormalization(momentum=0.8))
gen_model.add(Dense(512))
gen_model.add(LeakyReLU(alpha=0.2))
gen_model.add(BatchNormalization(momentum=0.8))
gen_model.add(Dense(1024))
gen_model.add(LeakyReLU(alpha=0.2))
gen_model.add(BatchNormalization(momentum=0.8))
gen_model.add(Dense(np.prod(img_shape), activation='tanh'))
gen_model.add(Reshape(img_shape))

gen_model.summary()

gen_noise = Input(shape=(latent_dim,))
gen_img = gen_model(gen_noise)

generator = Model(gen_noise, gen_img)

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# --------------------------------------------------

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# The combined model (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

combined.summary()

# --------------------------------------------------

epochs=30000
batch_size=32

# --------------------------------------------------

# Rescale -1 to 1
X_train_scaled = X_train_raw / 127.5 - 1.
X_train = np.expand_dims(X_train_scaled, axis=3)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# --------------------------------------------------

for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    gen_noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generate a batch of new images
    gen_imgs = generator.predict(gen_noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (to have the discriminator label samples as valid)
    g_loss = combined.train_on_batch(noise, valid)

    # Plot the progress
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    
# --------------------------------------------------

# Print samples of what the generator gives us
r, c = 5, 5
img_noise = np.random.normal(0, 1, (r * c, latent_dim))
pred_gen_imgs = generator.predict(img_noise)

# Rescale images 0 - 1
pred_gen_imgs = 0.5 * pred_gen_imgs + 0.5

fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(pred_gen_imgs[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
plt.show()
plt.close()
