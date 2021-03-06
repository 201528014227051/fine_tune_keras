import pickle
import random

import h5py
import numpy as np
import pandas as pd

class Generator():
    """ Data generator to the neural image captioning model (NIC).
    The flow method outputs a list of two dictionaries containing
    the inputs and outputs to the network.
    # Arguments:
        data_path = data_path to the preprocessed data computed by the
            Preprocessor class.
    """
    def __init__(self,data_path='preprocessed_data/',
                 training_filename=None,
                 validation_filename=None,
                 image_features_filename=None,
                 voice_features_filename=None,
                 batch_size=100,
                 voice_dim = 6500,
                 class_path = None):

        self.data_path = data_path
        self.class_number = self.get_class_number(class_path)
        if training_filename == None:
            self.training_filename = data_path + 'training_data.txt'
        else:
            self.training_filename = self.data_path + training_filename


        if validation_filename == None:
            self.validation_filename = data_path + 'validation_data.txt'
        else:
            self.validation_filename = self.data_path + validation_filename

        if image_features_filename == None:
            self.image_features_filename = (data_path +
                                            'vgg16_image_content.h5')
        else:
            self.image_features_filename = data_path + image_features_filename

        if voice_features_filename == None:
            self.voice_features_filename = (data_path +
                                            'image_name_to_voice_features_word.h5')
        else:
            self.voice_features_filename = data_path + voice_features_filename

        self.dictionary = None
        self.training_dataset = None
        self.validation_dataset = None
        self.image_names_to_features = None

        data_logs = np.genfromtxt(self.data_path + 'data_parameters.log',
                                  delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))

        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.BOS = str(data_logs['BOS:'])
        self.EOS = str(data_logs['EOS:'])
        self.PAD = str(data_logs['PAD:'])
        self.VOCABULARY_SIZE = None
        self.word_to_id = None
        self.id_to_word = None
        self.BATCH_SIZE = batch_size
        self.num_voice_features = voice_dim
        self.image_class_lable ={}
        self.load_dataset()
        self.load_vocabulary()
        self.load_image_features()
        self.load_class_labels(class_path)

    def get_class_number(self, class_path):
        import os
        files = os.listdir(class_path)
        return len(files)
        
    def load_class_labels(self, class_path):
        import os
        files = os.listdir(class_path)
        i = 0
        for file in files:
            fp = open(os.path.join(class_path, file))
            while True:
                line = fp.readline().strip()
                if not line:
                    i += 1
                    break
                self.image_class_lable[line] = i
            fp.close()
    
    def load_vocabulary(self):
        print('Loading vocabulary...')
        word_to_id = pickle.load(open(self.data_path + 'word_to_id.p', 'rb'))
        id_to_word = pickle.load(open(self.data_path + 'id_to_word.p', 'rb'))
        self.VOCABULARY_SIZE = len(word_to_id)
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def load_image_features(self):
        self.image_names_to_features = h5py.File(
                                        self.image_features_filename, 'r')

    def load_voice_features(self):
        self.image_names_to_voice_features = h5py.File(self.voice_features_filename, 'r')

    def load_dataset(self):

        print('Loading training dataset...')
        train_data = pd.read_table(self.training_filename, delimiter='*')
        train_data = np.asarray(train_data,dtype=str)
        self.training_dataset = train_data

        print('Loading validation dataset...')
        validation_dataset = pd.read_table(
                                self.validation_filename,delimiter='*')
        validation_dataset = np.asarray(validation_dataset, dtype=str)
        self.validation_dataset = validation_dataset

    def return_dataset(self, path=None, dataset_name='all', mode='training'):
        print('Loading dataset in memory...')
        if path == None:
            path = self.data_path
        if mode == 'training':
            data = pd.read_table(self.training_filename, sep='*')
        elif mode == 'test':
            data = pd.read_table(path + 'test_data.txt', sep='*')
        if dataset_name != 'all':
            data = data[data['image_names'].str.contains(dataset_name)]

        data = np.asarray(data)
        data_size = data.shape[0]
        image_names = data[:, 0]
        image_features = np.zeros((data_size,self.MAX_TOKEN_LENGTH,
                                   self.IMG_FEATS))
        image_captions = np.zeros((data_size,self.MAX_TOKEN_LENGTH,
                                   self.VOCABULARY_SIZE))
        target_captions = np.zeros((data_size,self.MAX_TOKEN_LENGTH,
                                   self.VOCABULARY_SIZE))

        for image_arg, image_name in enumerate(image_names):
            caption = data[image_arg,1]
            one_hot_caption = self.format_to_one_hot(caption)
            image_captions[image_arg, :, :] = one_hot_caption
            target_captions[image_arg, :, :] = self.get_one_hot_target(
                                                            one_hot_caption)
            image_features[image_arg, :, :] = self.get_image_features(
                                                            image_name)

        return image_features, image_captions, target_captions,image_names

    def flow(self, mode):

        if mode == 'train':
            data = self.training_dataset
            #random.shuffle(data) #this is probably correct but untested 
        if mode == 'validation':
            data = self.validation_dataset

        image_names = data[:,0].tolist()
        empty_batch = self.make_empty_batch()
        images_batch = empty_batch[0]
        target_class_batch = empty_batch[1]


        batch_counter = 0
        while True:
            for data_arg, image_name in enumerate(image_names):

                images_batch[batch_counter, :]   = self.get_image_features(
                                                            image_name)
                target_class_batch[batch_counter, :] = self.get_label(image_name)


                if batch_counter == self.BATCH_SIZE - 1:
                    yield_dictionary = self.wrap_in_dictionary(images_batch,target_class_batch)
                    yield yield_dictionary

                    empty_batch = self.make_empty_batch()
                    images_batch = empty_batch[0]
                    target_class_batch = empty_batch[1]
                    batch_counter = 0

                batch_counter = batch_counter + 1

    def make_test_input(self,image_name=None):

        if image_name == None:
            image_name = random.choice(self.training_dataset[:, 0].tolist())

        one_hot_caption = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                        self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        one_hot_caption[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, :, :] = self.get_image_features(image_name)
        return one_hot_caption, image_features, image_name

    def make_empty_batch(self):
        images_batch = np.zeros((self.BATCH_SIZE, 224,224,3))
        target_class_batch = np.zeros((self.BATCH_SIZE,  self.class_number))
        return  images_batch , target_class_batch


    def format_to_one_hot(self,caption):
        tokenized_caption = caption.split()
        tokenized_caption = [self.BOS] + tokenized_caption + [self.EOS]
        one_hot_caption = np.zeros((self.MAX_TOKEN_LENGTH))
        word_ids = [self.word_to_id[word] for word in tokenized_caption
                        if word in self.word_to_id]
        for sequence_arg, word_id in enumerate(word_ids):
            one_hot_caption[sequence_arg] = word_id####
        return one_hot_caption

    def format_to_one_hot_origin(self,caption):
        tokenized_caption = caption.split()
        tokenized_caption = [self.BOS] + tokenized_caption + [self.EOS]
        one_hot_caption = np.zeros((self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        word_ids = [self.word_to_id[word] for word in tokenized_caption
                        if word in self.word_to_id]
        for sequence_arg, word_id in enumerate(word_ids):
            one_hot_caption[sequence_arg, word_id] = 1####
        return one_hot_caption

    def get_image_features(self, image_name):
        image_features = self.image_names_to_features[image_name]['image_features'][:]
        return image_features
    
    def get_voice_features(self, image_name):
        voice_features = self.image_names_to_voice_features[str(image_name)]\
                                            ['voice_features'][:]
        voice_input = np.zeros((self.MAX_TOKEN_LENGTH, self.num_voice_features))
        voice_input[0,:] = voice_features
        return voice_input
    
    def get_one_hot_target(self,one_hot_caption):
        one_hot_target = np.zeros_like(one_hot_caption)
        one_hot_target[:-1, :] = one_hot_caption[1:, :]  ##
        return one_hot_target

    def get_label(self, image_name):
        id = self.image_class_lable[image_name]
        output =  np.zeros((self.class_number))
        output[id] = 1
        return output

    def wrap_in_dictionary(self,image_features,
                           one_hot_target):

        return [{'input_1': image_features},
                {'output31': one_hot_target}]


