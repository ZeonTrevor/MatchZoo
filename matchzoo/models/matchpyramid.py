# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
from keras.optimizers import Adam
from matchzoo.models.model import BasicModel
from matchzoo.layers.DynamicMaxPooling import *
from matchzoo.utils.utility import *


class MatchPyramid(BasicModel):
    def __init__(self, config):
        super(MatchPyramid, self).__init__(config)
        self.__name = 'MatchPyramid'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MatchPyramid] parameter check wrong')
        print('[MatchPyramid] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', [3, 3])
        self.set_default('dpool_size', [3, 10])
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        def xor_match(x):
            t1 = x[0]
            t2 = x[1]
            t1_shape = t1.get_shape()
            t2_shape = t2.get_shape()
            t1_expand = K.tf.stack([t1] * t2_shape[1], 2)
            t2_expand = K.tf.stack([t2] * t1_shape[1], 1)
            out_bool = K.tf.equal(t1_expand, t2_expand)
            out = K.tf.cast(out_bool, K.tf.float32)
            out = K.tf.expand_dims(out, 3)
            return out

        def gaussian_kernel_match(x):
            t1 = x[0]
            t2 = x[1]
            t1_shape = t1.get_shape()
            t2_shape = t2.get_shape()
            t1_expand = K.tf.stack([t1] * t2_shape[1], 2)
            t2_expand = K.tf.stack([t2] * t1_shape[1], 1)
            t_diff = K.tf.subtract(t1_expand, t2_expand)
            t_diff_norm = K.tf.norm(t_diff, ord='euclidean', axis=3)  # L2-norm when input is a matrix and axis>0
            out = K.tf.exp(K.tf.negative(K.tf.square(t_diff_norm)))
            # out = K.tf.expand_dims(out, 3)
            return out

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')
        show_layer_info('Input', dpool_index)

        if self.config['similarity'] in ['dot', 'cosine', 'gaussian']:
            embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
            q_embed = embedding(query)
            show_layer_info('Embedding', q_embed)
            d_embed = embedding(doc)
            show_layer_info('Embedding', d_embed)

        if self.config['similarity'] == 'dot':
            cross = Dot(axes=[2, 2], normalize=False)([q_embed, d_embed])
            show_layer_info('Dot', cross)

        if self.config['similarity'] == 'cosine':
            cross = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
            show_layer_info('Cosine', cross)

        if self.config['similarity'] == 'indicator':
            cross = Lambda(xor_match)([query, doc])
            show_layer_info('Indicator', cross)

        if self.config['similarity'] == 'gaussian':
            cross = Lambda(gaussian_kernel_match)([q_embed, d_embed])
            show_layer_info('Gaussian', cross)

        cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(cross)
        show_layer_info('Reshape', cross_reshape)

        conv2d = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])
        # maxpool = MaxPooling2D(pool_size=(self.config['dpool_size'][0], self.config['dpool_size'][1]), padding="valid")

        conv1 = conv2d(cross_reshape)
        show_layer_info('Conv2D', conv1)
        pool1 = dpool([conv1, dpool_index])
        show_layer_info('DynamicMaxPooling', pool1)
        # pool1 = maxpool(conv1)
        # show_layer_info('MaxPooling2D', pool1)
        pool1_flat = Flatten()(pool1)
        show_layer_info('Flatten', pool1_flat)
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        show_layer_info('Dropout', pool1_flat_drop)

        dense1 = Dense(128, activation='relu')(pool1_flat_drop)
        show_layer_info('Dense', dense1)

        # dense2 = Dense(128, activation='relu')(dense1)
        # show_layer_info('Dense', dense2)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            # out_ = Dense(1)(pool1_flat_drop)
            out_ = Dense(1)(dense1)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        # model = Model(inputs=[query, doc], outputs=out_)
        return model
