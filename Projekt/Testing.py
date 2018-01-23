"""


@author: Yannic
"""

import os, errno

# try:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils import plot_model
from keras import backend as K


matplotlib.use('Agg')

# Hyper Parameters:
LR = 0.001  # 0.00013
DECAY = 0.0001
KERNEL_SIZE = (4, 4)
FILTER = 64
# dimensions of our images.
img_width, img_height = 300, 180

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

labels = sorted([drtr for drtr in os.listdir(train_data_dir) if not drtr.title().startswith('.')])
# print labels

nr_of_classes = len(os.listdir(train_data_dir)) - 1
# print nr_of_classes
nb_train_samples = 349  # 32 * nr_of_classes
nb_validation_samples = 188  # 18 * nr_of_classes
epochs = 42
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = None
# def train_model():

model = Sequential()

model.add(Conv2D(FILTER, KERNEL_SIZE, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(FILTER, KERNEL_SIZE, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(FILTER, KERNEL_SIZE, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(FILTER, KERNEL_SIZE, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(FILTER, KERNEL_SIZE, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nr_of_classes))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=LR, rho=0.9, epsilon=K.epsilon(), decay=DECAY),
              metrics=['categorical_accuracy'])  # ['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=45,
    rescale=1. / 255,
    shear_range=0.17,
    zoom_range=0.17,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)


# train_model()


def predict_image(path):
    test_model = load_model('testModel.h5')
    img = load_img(path, False, target_size=(img_width, img_height))
    print "\n\nTrying to predict following image (" + path + "): "
    # plt.imshow(img)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print "The image probably shows the flag of " + labels[test_model.predict_classes(x, verbose=0)[0]] + "."
    prediction = test_model.predict(x, verbose=0)
    classes = test_model.predict_classes(x, verbose=0)
    probs = test_model.predict_proba(x, verbose=0)
    print "Prediction | Classes | Probabilities"
    print prediction, '\n', classes, '\n', probs


predict_image("beats.png")
predict_image("germanyTest.png")
predict_image("usaTest.png")
predict_image("us-russia-flag.png")
predict_image("usaPillow.png")
predict_image("data/train/usa/usa25.png")


def augment_image(country, nr):
    datagen = ImageDataGenerator(
        rotation_range=17,
        rescale=1. / 255,
        shear_range=0.07,
        zoom_range=0.07,
        horizontal_flip=True,
        fill_mode='constant',
        cval=128)
    # ImageDataGenerator(
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest')
    img = load_img('data/train/'+country+'/'+country+nr+'.png')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='augmented-images', save_prefix=country, save_format='png'):
        i += 1
        if i > 20:
            break


# augment_image('germany', '11')


# except OSError:
#     print ":("


# plot_model(model, to_file='model.png')  # install pydot and graphviz for `pydotprint` to work
# model.save('testModel.h5')
