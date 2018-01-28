"""


@author: Yannic
"""

import os, errno

# try:

import numpy as np
from scipy import misc
from sklearn import metrics
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


# matplotlib.use('Agg')

img_dir = 'data/'

# Hyper Parameters:
LR = 0.001  # 0.00013
MOMENTUM = 0.9
DECAY = 1e-4
KERNEL_SIZE = (4, 4)
FILTER = 64
RMSPROP = RMSprop(lr=LR, rho=0.9, epsilon=K.epsilon(), decay=DECAY)
SGD_OPT = SGD(lr=LR, momentum=MOMENTUM, decay=DECAY, nesterov=True)

temp_dir = img_dir + 'train/canada/'
img = misc.imread(temp_dir + os.listdir(temp_dir)[1])
img_shape = img.shape
img_width, img_height = img_shape[1], img_shape[0]

train_data_dir = img_dir+'train'
validation_data_dir = img_dir+'validation'

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


def train_model(from_scratch, nr_convlayer=1):
    global model
    if from_scratch:
        model = Sequential()

        for _ in range(nr_convlayer + 1):
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
                      optimizer=RMSPROP,
                      metrics=['categorical_accuracy'])  # ['accuracy'])

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rotation_range=45,
            rescale=1. / 255,
            shear_range=0.17,
            zoom_range=0.17,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        test_datagen = ImageDataGenerator(rescale=1. / 255)  # only rescaling

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

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        plot_training(history)
    else:
        model = load_model('testModel.h5')


def predict_image(path):
    img = load_img('predict/'+path, False, target_size=(img_width, img_height))
    print "\n\nTrying to predict following image (" + path + "): "
    # plt.imshow(img)
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    print "The image probably shows the flag of " + labels[model.predict_classes(x, verbose=0)[0]] + "."
    prediction = model.predict(x, verbose=0) * 100
    classes = model.predict_classes(x, verbose=0)
    probs = model.predict_proba(x, verbose=0) * 100
    print "Prediction | Classes | Probabilities"
    np.set_printoptions(linewidth=128, formatter={'float': '{: 0.3f}'.format})
    print prediction, '\n', classes, '\n', probs


def predict():
    for img in os.listdir('predict'):
        predict_image(img)


def augment_image(country, nr):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.07,
        height_shift_range=0.07,
        shear_range=0.13,
        zoom_range=0.13,
        horizontal_flip=True,
        fill_mode='nearest',
        cval=128)
    img = load_img('data/train/'+country+'/'+country+nr+'.png')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='augmented-images', save_prefix=country, save_format='png'):
        i += 1
        if i > 9:
            break


def calculate_metrics():
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
            'data/test',
            target_size=(img_width, img_height),
            batch_size=10,
            class_mode=None,  # only data, no labels
            shuffle=False)  # keep data in same order as labels

    probabilities = model.predict_generator(generator, 50)

    y_labels = []
    for i in range(10):
        y_labels += [i] * 50
    y_true = np.array(y_labels)
    y_pred = np.argmax(probabilities, axis=1)

    # print y_true
    # print y_pred

    print labels
    cm = metrics.confusion_matrix(y_true, y_pred)  # , labels=labels)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    report = metrics.classification_report(y_true, y_pred, list(range(10)), target_names=labels)

    print cm
    print "\nAccuracy:", accuracy, "| Precision:", precision
    print "\nReport:\n"
    print report


def plot_training(fit):
    print(fit.history.keys())
    # summarize history for accuracy
    plt.plot(fit.history['categorical_accuracy'])
    plt.plot(fit.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig('training-history.png')


train_model(from_scratch=1, nr_convlayer=3)

calculate_metrics()

# predict()

# augment_image('canada', '07')

# plot_model(model, to_file='model.png')  # install pydot and graphviz for `pydotprint` to work

# model.save('testModel.h5')


# except OSError:
#     print ":("
