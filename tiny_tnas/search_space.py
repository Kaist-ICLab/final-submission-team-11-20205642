import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, UpSampling1D
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, DepthwiseConv1D, SeparableConv1D, MaxPool1D, GlobalMaxPool1D, Flatten, Conv1DTranspose
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization

# SUPPORT_LAYERS = ['MLP', 'RNN', 'LSTM', 'GRU', 'CNN', 'Transformers']
SUPPORT_LAYERS = ['MLP', 'CNN']
NETWORK_PROBS = {
    'classification': [0.4515836567, 0.5484163433], # [0.1814512408, 0.1323490691, 0.1496907102, 0.1724838191, 0.2214311401, 0.1425940207],
    'regression': [0.505527877, 0.494472123], # [0.1719333474, 0.161527619, 0.1615566001, 0.1616595295, 0.1713138921, 0.1720090119],
    'anomaly_detection': [0.4603315571, 0.5396684429], # [0.151912337, 0.1534102936, 0.1749841249, 0.1898985623, 0.1780940131, 0.1517006692]
}
NUM_LAYERS = [2, 3, 4, 5, 6]
SUBSPACES = {
    'MLP': { 'units': [8, 16, 32, 64, 128, 256] },
    # 'RNN': { 'units': [8, 16, 32, 64, 128, 256], 'dropout': [0.0, 0.1, 0.2, 0.3, 0.4] },
    # 'LSTM': { 'units': [8, 16, 32, 64, 128, 256], 'dropout': [0.0, 0.1, 0.2, 0.3, 0.4] },
    # 'GRU': { 'units': [8, 16, 32, 64, 128, 256], 'dropout': [0.0, 0.1, 0.2, 0.3, 0.4] },
    'CNN': { 'filters': [8, 16, 32, 64, 128, 256], 'kernel_size': [2, 3, 5, 7, 9], 'strides': [1, 2, 3]},
    # 'Transformers': {'head_size': [4, 8, 16, 32, 64, 128], 'num_heads': [2, 4, 6, 8], 'ff_dim': [8, 16, 32], 'dropout': [0.0, 0.1, 0.2, 0.3, 0.4] }
}

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def get_layer(layer_name, **kwargs):
    if layer_name == 'MLP':
        return Dense(kwargs['units'], activation='relu')
    elif layer_name == 'RNN':
        return SimpleRNN(kwargs['units'], return_sequences=kwargs['return_sequences'], dropout=kwargs['dropout'])
    elif layer_name == 'LSTM':
        return LSTM(kwargs['units'], return_sequences=kwargs['return_sequences'], dropout=kwargs['dropout'])
    elif layer_name == 'GRU':
        return GRU(kwargs['units'], return_sequences=kwargs['return_sequences'], dropout=kwargs['dropout'])
    elif layer_name == 'CNN':
        return Conv1D(kwargs['filters'], kernel_size=kwargs['kernel_size'], strides=kwargs['strides'], padding='same', activation='relu')
    elif layer_name == 'Transformers':
        return transformer_encoder


def get_random_architectures(task, num_archs, network_type, allow_mix):
    # return a set of `num_archs` architecture configurations
    
    archs = []
    selected_types = SUPPORT_LAYERS if network_type == 'ALL' else network_type
    assert sum([l in SUPPORT_LAYERS for l in selected_types]) == len(selected_types) # check whether all the given layer types are correctly formatted
    
    while len(archs) < num_archs:
        # maximun size is `number of layer` x `number of configs / layer`
        # first value of each row indicates `layer type`
        arch = np.zeros([6, 5], dtype=object)
        if task == 'anomaly_detection':
            arch = np.zeros([2, 6, 5], dtype=object)
        n_layers = np.random.choice(NUM_LAYERS, size=1)[0]

        init_layer_type = None
        for l in range(n_layers):
            # randomly (re-)select the layer type only when a combination is allowed (i.e., select for each layer) 
            # or at the first layer
            if allow_mix or l == 0:
                if network_type == 'ALL':
                    # re-weight the layer type probability by the given task
                    layer_type = np.random.choice(selected_types, size=1, p=NETWORK_PROBS[task])[0]
                    init_layer_type = layer_type
                else:
                    # uniform selection for the given layer types
                    layer_type = np.random.choice(selected_types, size=1)[0]
                    init_layer_type = layer_type
            else:
                layer_type = init_layer_type

            if task == 'anomaly_detection':
                arch[0][l][0] = layer_type # encoder network
                arch[1][-l-1][0] = layer_type # decoder network
            else:
                arch[l][0] = layer_type
            
            for idx, op in enumerate(SUBSPACES[layer_type].keys()):
                if task == 'anomaly_detection':
                    arch[0][l][idx + 1] = np.random.choice(SUBSPACES[layer_type][op], size=1)[0]
                    arch[1][-l-1][idx + 1] = np.random.choice(SUBSPACES[layer_type][op], size=1)[0]
                else:
                    arch[l][idx + 1] = np.random.choice(SUBSPACES[layer_type][op], size=1)[0]
        
        archs.append(arch)
        
    return np.array(archs) # [number of archs, number of layers, number of operations]



def get_predefined_mlps(task, x_train, y_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    if task == 'classification':
        n_classes = np.unique(y_train).shape[0]
        
        inputs = Input(shape=[seq_length, n_features])
        x = Flatten()(inputs)
        x = Dense(128//n_features, activation="relu")(x)
        x = Dense(128//n_features, activation="relu")(x)
        x = Dense(128//n_features, activation="relu")(x)
        outputs = Dense(n_classes, activation="softmax")(x)

    elif task == 'regression':
        inputs = Input(shape=[seq_length, n_features])
        x = Flatten()(inputs)
        x = Dense(128, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(1, activation='linear')(x)

    elif task == 'anomaly_detection':
        inputs = Input(shape=[seq_length, n_features])
        x = Flatten()(inputs)
        x = Dense(128, activation="relu")(x)
        x = Dense(96, activation="relu")(x)
        encoder_output = Dense(32, activation="relu")(x)
        
        x = RepeatVector(seq_length)(encoder_output)
        x = Dense(96, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        outputs = TimeDistributed(Dense(n_features))(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=f'MLP-{task}')

        
def get_predefined_rnns(task, x_train, y_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    if task == 'classification':
        n_classes = np.unique(y_train).shape[0]
        
        inputs = Input(shape=[seq_length, n_features])
        x = SimpleRNN(128, return_sequences=True)(inputs)
        x = SimpleRNN(128)(x)
        outputs = Dense(n_classes, activation="softmax")(x)

    elif task == 'regression':
        inputs = Input(shape=[seq_length, n_features])
        x = SimpleRNN(128, return_sequences=True)(inputs)
        x = SimpleRNN(128)(x)
        outputs = Dense(1, activation='linear')(x)

    elif task == 'anomaly_detection':
        inputs = Input(shape=[seq_length, n_features])
        x = SimpleRNN(128, return_sequences=True)(inputs)        
        x = SimpleRNN(64)(x)
        encoder_output = Dense(32, activation="relu")(x)
        
        x = RepeatVector(seq_length)(encoder_output)
        x = SimpleRNN(64, return_sequences=True)(x)
        x = SimpleRNN(128, return_sequences=True)(x)
        outputs = TimeDistributed(Dense(n_features))(x)        

    return keras.Model(inputs=inputs, outputs=outputs, name=f'RNN-{task}')

def get_predefined_lstm(task, x_train, y_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    if task == 'classification':
        n_classes = np.unique(y_train).shape[0]
        
        inputs = Input(shape=[seq_length, n_features])
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(64)(x)
        outputs = Dense(n_classes, activation="softmax")(x)

    elif task == 'regression':
        inputs = Input(shape=[seq_length, n_features])
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(64)(x)
        outputs = Dense(1, activation='linear')(x)

    elif task == 'anomaly_detection':
        inputs = Input(shape=[seq_length, n_features])
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32)(x)
        encoder_output = Dense(16, activation="relu")(x)
                
        x = RepeatVector(seq_length)(encoder_output)
        x = LSTM(32, return_sequences=True)(x)
        x = LSTM(64, return_sequences=True)(x)
        outputs = TimeDistributed(Dense(n_features))(x)        

    return keras.Model(inputs=inputs, outputs=outputs, name=f'LSTM-{task}')

def get_predefined_gru(task, x_train, y_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    if task == 'classification':
        n_classes = np.unique(y_train).shape[0]
        
        inputs = Input(shape=[seq_length, n_features])
        x = GRU(74, return_sequences=True)(inputs)
        x = GRU(74)(x)
        outputs = Dense(n_classes, activation="softmax")(x)

    elif task == 'regression':
        inputs = Input(shape=[seq_length, n_features])
        x = GRU(74, return_sequences=True)(inputs)
        x = GRU(74)(x)
        outputs = Dense(1, activation='linear')(x)

    elif task == 'anomaly_detection':
        inputs = Input(shape=[seq_length, n_features])
        x = GRU(72, return_sequences=True)(inputs)
        x = GRU(32)(x)
        encoder_output = Dense(16, activation="relu")(x)
                
        x = RepeatVector(seq_length)(encoder_output)
        x = GRU(32, return_sequences=True)(x)
        x = GRU(72, return_sequences=True)(x)
        outputs = TimeDistributed(Dense(n_features))(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=f'GRU-{task}')


def get_predefined_cnns(task, x_train, y_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    if task == 'classification':
        n_classes = np.unique(y_train).shape[0]
        
        inputs = Input(shape=[seq_length, n_features])
        x = Conv1D(128, kernel_size=7, activation='relu', padding='same')(inputs)
        x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
        x = MaxPool1D(2)(x)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPool1D(2)(x)
        x = Flatten()(x)
        outputs = Dense(n_classes, activation="softmax")(x)

    elif task == 'regression':
        inputs = Input(shape=[seq_length, n_features])
        x = Conv1D(128, kernel_size=7, activation='relu', padding='same')(inputs)
        x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
        x = MaxPool1D(2)(x)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPool1D(2)(x)
        x = Flatten()(x)
        outputs = Dense(1, activation='linear')(x)

    elif task == 'anomaly_detection':
        inputs = Input(shape=[seq_length, n_features])
        x = Conv1D(128, kernel_size=7, activation='relu', padding='same')(inputs)
        x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
        encoder_output = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
        
        x = Conv1DTranspose(16, kernel_size=3, activation='relu', padding='same')(encoder_output)
        x = Conv1DTranspose(32, kernel_size=5, activation='relu', padding='same')(x)
        x = Conv1DTranspose(128, kernel_size=7, activation='relu', padding='same')(x)        
        outputs = Conv1DTranspose(1, kernel_size=3, padding='same')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=f'CNN-{task}')


def get_predefined_transformers(task, x_train, y_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    if task == 'classification':
        n_classes = np.unique(y_train).shape[0]
        
        inputs = Input(shape=[seq_length, n_features])
        x = inputs
        for _ in range(3):
            x = transformer_encoder(x, 256//n_features, 8, 8, 0.1)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(n_classes, activation="softmax")(x)

    elif task == 'regression':
        inputs = Input(shape=[seq_length, n_features])
        x = inputs
        for _ in range(3):
            x = transformer_encoder(x, 256//n_features, 8, 8, 0.1)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='linear')(x)

    elif task == 'anomaly_detection':
        inputs = Input(shape=[seq_length, n_features])
        x = inputs
        for _ in range(3):
            x = transformer_encoder(x, 32, 8, 8, 0.1)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.2)(x)        
        encoder_output = Dense(16, activation="relu")(x)
                
        x = encoder_output
        for _ in range(3):
            x = transformer_encoder(x, 32, 8, 8, 0.1)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(n_features)(x)    

    return keras.Model(inputs=inputs, outputs=outputs, name=f'Transformers-{task}')