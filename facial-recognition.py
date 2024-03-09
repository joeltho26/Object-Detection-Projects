import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

train_path = 'data/facial/train'
val_path = 'data/facial/val'

vgg = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])

for layer in vgg.layers:
    layer.trainable = False
    
folders = glob('data/facial/train/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255., 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255.)

training_set = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_set = test_datagen.flow_from_directory(val_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

r = model.fit_generator(training_set, validation_data=test_set, epochs=EPOCHS, steps_per_epoch=len(training_set), validation_steps=len(test_set))

plt.plot(r.history('loss'), label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history('accuracy'), label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

model.save('face_recognition_model.h5')
