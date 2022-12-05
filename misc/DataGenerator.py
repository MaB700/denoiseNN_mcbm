import numpy as np
import tensorflow as tf
#import keras


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, tree, batch_size=32, input_size=(72, 32, 1), shuffle=True):
        'Initialization'
        self.tree = tree
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.tree.num_entries / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if index*self.batch_size >= self.big_batch_size*self.big_batch_index:
            self.__get_big_batch()
        #if index switches to i'th big_batch -> load new big batch from file/disk
        #do reshape and so on in here
        x, y = self.__data_generation(self.batch_index*self.batch_size, (self.batch_index+1)*self.batch_size)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.list_IDs))
        #if self.shuffle == True:
        #    np.random.shuffle(self.indexes)

    def __data_generation(self, lower_index, upper_index):
        'Generates data containing batch_size samples' 
        # load from data on ram
        x = self.x_big[lower_index:upper_index]
        y = self.y_big[lower_index:upper_index]
        self.batch_index += 1
        return x, y

    def __get_big_batch(self):
        self.x_big = tf.reshape(self.tree["time"].array()[self.big_batch_index*self.big_batch_size:(self.big_batch_index+1)*self.big_batch_size], [-1, 72, 32, 1])
        self.y_big = tf.reshape(self.tree["tar"].array()[self.big_batch_index*self.big_batch_size:(self.big_batch_index+1)*self.big_batch_size], [-1, 72, 32, 1])
        self.big_batch_index += 1
        self.batch_index = 0
        