import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
class Evaluator(object):

    def __init__(self, model,
            data_path='preprocessed_data/',
            images_path='iaprtc12/',
            voice_dim = 6500,
            log_filename='data_parameters.log',
            test_data_filename='test_data.txt',
            word_to_id_filename='word_to_id.p',
            id_to_word_filename='id_to_word.p',
            image_name_to_features_filename='vgg16_image_content.h5',
            class_path = None,):
        self.class_path = class_path
        self.model = model
        self.data_path = data_path
        self.images_path = images_path
        self.log_filename = log_filename
        data_logs = self._load_log_file()
        self.BOS = str(data_logs['BOS:'])
        self.EOS = str(data_logs['EOS:'])
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.voice_feats = voice_dim
        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        self.test_data = pd.read_table(data_path +
                                       test_data_filename, sep='*')
        self.word_to_id = pickle.load(open(data_path +
                                           word_to_id_filename, 'rb'))
        self.id_to_word = pickle.load(open(data_path +
                                           id_to_word_filename, 'rb'))
        self.VOCABULARY_SIZE = len(self.word_to_id)
        self.image_names_to_features = h5py.File(data_path +
                                        image_name_to_features_filename)
        self.image_class_lable = {}
        self._load_class_labels()
 

    def _load_log_file(self):
        data_logs = np.genfromtxt(self.data_path + 'data_parameters.log',
                                  delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))
        return data_logs

    def _load_class_labels(self):
        import os
        files = os.listdir(self.class_path)
        i = 0
        for file in files:
            fp = open(os.path.join(self.class_path, file))
            while True:
                line = fp.readline().strip()
                if not line:
                    i += 1
                    break
                self.image_class_lable[line] = i
            fp.close()

    def is_right_cla(self, res_cla, image_name):
        dict = {}
        for res in res_cla:
            if res in dict:
                dict[res] += 1
            else:
                dict[res] = 1
        dict_reverse = {}
        for key, value in dict.items():
            dict_reverse[value] = key
        res_once = list(set(res_cla))
        max_count = 0
        for res in res_once:
            if dict[res]>max_count:
                max_count = dict[res]
        if dict_reverse[max_count] == self.image_class_lable[image_name]:
            return True
        else:
            return False

    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'rsicd_mt_05_10_direct.txt'

        image_names = self.test_data['image_names'].tolist()
        test_number_all = len(image_names)
        cla_right_count = 0
        for arg_voice, image_name in enumerate(image_names):
            print(image_name)
            res_cla = []
            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            image_features = np.zeros((1, 224, 224, 3))
            image_features[0, :] = features
            neural_caption = []
            clas_single = self.model.predict(image_features)
            cla_id = np.argmax(clas_single[0, :])
            res_cla.append(cla_id)
            #if self.is_right_cla(res_cla, image_name):
            if cla_id == self.image_class_lable[image_name]:
                cla_right_count += 1
        print('right:%d'%cla_right_count)
        print('all:%d'%test_number_all)
        acc = cla_right_count/test_number_all
        print('classification_accuracy:%f'%acc)

if __name__ == '__main__':
    from keras.models import load_model

    root_path = '../datasets/rsicd/'
    data_path = root_path 
    images_path = '/home/user2/qubo_captions/data/RSICD/imgs/'
    model_filename = '../trained_models/rsicd_finetune_vgg16/rsicd_weights.95-0.81_finetune.hdf5'
    model = load_model(model_filename)
    evaluator = Evaluator(model, data_path, images_path, class_path = './txtclasses_rsicd/')
    evaluator.write_captions()
