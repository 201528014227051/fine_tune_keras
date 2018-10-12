from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from generator import Generator
from data_manager import DataManager
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
num_epochs = 100
batch_size = 10#100#256
voice_dim = 300#1690#
root_path = '../datasets/sydney/'
class_txt_path = './txtclasses_sydney/'
captions_filename = root_path + 'complete_data.txt'
data_manager = DataManager(data_filename=captions_filename,
                            max_caption_length=30,
                            word_frequency_threshold=2,
                            extract_image_features=True,
                            extract_voice_features=False,
                            cnn_extractor='vgg16',
                            image_directory='/home/user2/qubo_captions/data/Sydney/imgs/',
                            voice_directory = root_path + 'filter_word_voice0/',
                            split_data=False,
                            dump_path=root_path,
                            voice_dim = voice_dim)

data_manager.preprocess()
# 

# create the base pre-trained model
def construct_model():
    base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu', name = 'fea1024')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(7, activation='softmax', name = 'output21')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = construct_model()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
    # layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
#model.fit_generator(...)
generator = Generator(data_path=root_path,
                      batch_size=batch_size,
                      voice_dim = voice_dim,
                      image_features_filename = 'vgg16_image_content.h5',
                      voice_features_filename = 'image_name_to_voice_features_word_pca300.h5',
                      class_path = class_txt_path)

num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:18]:
   layer.trainable = False
for layer in model.layers[18:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
#csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('../trained_models/sydney_finetune_vgg16/' +
               'rsicd_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [ model_checkpoint, reduce_learning_rate]

model.fit_generator(generator=generator.flow(mode='train'),
                    steps_per_epoch=int(num_training_samples / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=int(num_validation_samples / batch_size))