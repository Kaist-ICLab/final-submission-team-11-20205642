import sys, os
import tensorflow as tf

from tiny_tnas.converters import convert_model
from mltk.core.tflite_micro import TfliteMicro


def get_model_profile(model, allow_mix, layer_type):
    tflite_model = convert_model(model)
    open('temp_model.tflite', "wb").write(tflite_model)
    
    model_profiles = {
        'model_size': 0,
        'runtime_memory': 0,
        'flops': 0,
        'macs': 0,
        'inference_time': 0,
        'energy': 0,
        'params': model.count_params()
    }
    
    if allow_mix:
        model_profiles['model_size'] = get_model_size(tflite_model)
    else:
        profile = TfliteMicro.profile_model('temp_model.tflite', return_estimates=True)
        model_profiles['model_size'] = profile.flatbuffer_size / 1000.0
        model_profiles['runtime_memory'] = profile.runtime_memory_bytes / 1000.0
        model_profiles['flops'] = profile.ops
        model_profiles['macs'] = profile.macs
        model_profiles['inference_time'] = profile.time
        model_profiles['energy'] = profile.energy
            
    os.remove('temp_model.tflite')
    return model_profiles

def get_model_size(tflite_model):
    size_in_bytes = sys.getsizeof(tflite_model)
    return size_in_bytes / 1000.0