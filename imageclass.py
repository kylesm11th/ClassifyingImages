# Importing all necessary libraries
# Every image in the dataset is of the size 224*224.
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 227, 227


train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 10
batch_size = 16

#

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# This part is to check the data format i.e the RGB channel is coming first or last

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Conv2D is the layer to convolve the image into multiple images
# Activation is the activation function.
# MaxPooling2D is used to max pool the value from the given size matrix and same is used for the next 2 layers.
# then, Flatten is used to flatten the dimensions of the image obtained after convolving it.
# Dense is used to make this a fully connected model and is the hidden layer.
# Dropout is used to avoid overfitting on the dataset.
# Dense is the output layer contains only one neuron which decide to which category image belongs.

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Compile function is used here that involve the use of loss, optimizers and metrics.

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# Now, the part of dataGenerator comes into the figure

model.save('model_saved.h5')

# At last, we can also save the model.


model = load_model('model_saved.h5')

image = load_img('v_data/validation/planes/5.jpg', target_size=(227, 227))
img = np.array(image)
img = img / 255.0
img = img.reshape([1, 227, 227, 3])
label = model.predict_step(img)
print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])
