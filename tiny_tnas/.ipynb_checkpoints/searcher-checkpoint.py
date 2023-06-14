import gc, os

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, UpSampling1D
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, DepthwiseConv1D, SeparableConv1D, MaxPool1D, GlobalMaxPool1D, Flatten, Conv1DTranspose
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization

from tiny_tnas.search_space import get_random_architectures, get_layer, SUPPORT_LAYERS
from tiny_tnas.profiler import get_model_profile
from tiny_tnas.zero_proxies import get_zico_score


class Searcher(object):
    def __init__(self, max_ram, max_flash, task, x_train, y_train, network_type='ALL'):
        self.max_ram = max_ram
        self.max_flash = max_flash
        self.network_type = network_type
        self.task = task
        self.x_train = x_train
        self.y_train = y_train
        self.curr_candidates = None

    def _get_layer_from_configs(self, configs, is_last):
        layer_type = configs[0]        
        if layer_type == 'MLP':
            return get_layer(layer_type, units=int(configs[1]))
        elif layer_type in ['RNN', 'LSTM', 'GRU']:
            return get_layer(layer_type, units=int(configs[1]), return_sequences=not is_last, dropout=int(configs[2]))
        elif layer_type == 'CNN':
            return get_layer(layer_type, filters=int(configs[1]), kernel_size=int(configs[2]), strides=int(configs[3]))
        elif layer_type == 'Transformers':
            return get_layer(layer_type)

    def _build_and_filter_models(self, allow_mix):
        runnable_candidates = []
        
        seq_length = self.x_train.shape[1]
        n_features = self.x_train.shape[2]

        if self.task == 'classification':
            n_classes = np.unique(self.y_train).shape[0]
            for cand in tqdm(self.curr_candidates):
                tf.keras.backend.clear_session()

                inputs = Input(shape=[seq_length, n_features])
                for l, config in enumerate(cand):
                    if config[0] in SUPPORT_LAYERS:
                        is_last = True if l == len(cand) - 1 else cand[l + 1][0] not in SUPPORT_LAYERS
                        l_layer = self._get_layer_from_configs(config, is_last)
                        if l == 0:
                            if config[0] == 'MLP' and allow_mix == False:
                                x = Flatten()(inputs)
                                x = l_layer(x) if config[0] != 'Transformers' else l_layer(inputs, config[1], config[2], config[3], config[4])
                            else:
                                x = l_layer(inputs) if config[0] != 'Transformers' else l_layer(inputs, config[1], config[2], config[3], config[4])
                        else:
                            x = l_layer(x) if config[0] != 'Transformers' else l_layer(x, config[1], config[2], config[3], config[4])
                x = Flatten()(x)
                outputs = Dense(n_classes, activation="softmax")(x)
                model = keras.Model(inputs=inputs, outputs=outputs)

                m_profile = get_model_profile(model, allow_mix, cand[0][0])
                if m_profile['model_size'] <= self.max_flash and m_profile['runtime_memory'] <= self.max_ram:
                    # get zico scores
                    m_profile['zico'] = get_zico_score(model, self.x_train, self.y_train, self.task)
                    runnable_candidates.append({ 'model': model, 'profile': m_profile })
                
                del model
                keras.backend.clear_session()
                gc.collect()
                
        elif self.task == 'regression':
            for cand in tqdm(self.curr_candidates):
                tf.keras.backend.clear_session()
                
                inputs = Input(shape=[seq_length, n_features])
                for l, config in enumerate(cand):
                    if config[0] in SUPPORT_LAYERS:
                        is_last = True if l == len(cand) - 1 else cand[l + 1][0] not in SUPPORT_LAYERS
                        l_layer = self._get_layer_from_configs(config, is_last)
                        if l == 0:
                            if config[0] == 'MLP' and allow_mix == False:
                                x = Flatten()(inputs)
                                x = l_layer(x) if config[0] != 'Transformers' else l_layer(inputs, config[1], config[2], config[3], config[4])
                            else:
                                x = l_layer(inputs) if config[0] != 'Transformers' else l_layer(inputs, config[1], config[2], config[3], config[4])
                        else:
                            x = l_layer(x) if config[0] != 'Transformers' else l_layer(x, config[1], config[2], config[3], config[4])
                x = Flatten()(x)
                outputs = Dense(1, activation='linear')(x)
                model = keras.Model(inputs=inputs, outputs=outputs)

                m_profile = get_model_profile(model, allow_mix, cand[0][0])
                if m_profile['model_size'] <= self.max_flash and m_profile['runtime_memory'] <= self.max_ram:
                    # get zico scores
                    m_profile['zico'] = get_zico_score(model, self.x_train, self.y_train, self.task)
                    runnable_candidates.append({ 'model': model, 'profile': m_profile })
                
                del model
                keras.backend.clear_session()
                gc.collect()
        
        return runnable_candidates

    
    def _rank_candidates(self, top_k, proxy):
        return topk_models

    def run_search(self, n_archs=1000, top_k=1, proxy='flops', method='random', allow_mix=False, batch_size=32):
        self.curr_candidates = get_random_architectures(self.task, n_archs, self.network_type, allow_mix)
        # filter runable models by memory and storage constraints
        self.curr_candidates = self._build_and_filter_models(allow_mix)

        if proxy == 'flops':
            self.curr_candidates = sorted(self.curr_candidates, key=lambda d: d['profile']['flops'], reverse=True)            
            return self.curr_candidates[:top_k]
        elif proxy == 'zico':
            self.curr_candidates = sorted(self.curr_candidates, key=lambda d: d['profile']['zico'], reverse=True)            
            return self.curr_candidates[:top_k]
        else:
            return self.curr_candidates[:top_k]
        