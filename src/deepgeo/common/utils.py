from tensorflow.python.client import device_lib


def check_dict_parameters(params, mandatory=[], default={}):
    for par in mandatory:
        if par not in params:
            raise AttributeError('Mandatory argument "raster_array" does not exists in parameters!')

    for par, value in default.items():
        if par not in params:
            params[par] = value

    return params


# Function bellow based on: https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
