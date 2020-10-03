import csv
import ast
from tensorflow.python.client import device_lib


def check_dict_parameters(params, mandatory=[], default={}):
    for par in mandatory:
        if par not in params:
            raise AttributeError('Mandatory argument "' + par + '" does not exists in parameters!')

    for par, value in default.items():
        if par not in params:
            params[par] = value

    return params


# Function bellow based on: https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def nested_list_contains(nl, target):
    for thing in nl:
        if type(thing) is list:
            if nested_list_contains(thing, target):
                return True
        if thing == target:
            return True
    return False


def read_csv_2_dict(csv_path, keys_exclude=[]):
    with open(csv_path, mode='r') as infile:
        reader = csv.reader(infile, delimiter=';')
        my_dict = {}
        for row in reader:
            if row[0] not in keys_exclude:
                key = row[0]
                try:
                    val = ast.literal_eval(row[1].strip('\n'))
                except:
                    val = row[1]
                my_dict[key] = val
        return my_dict


def save_dict_2_csv(dict_2_save, out_path, delimiter=';'):
    with open(out_path, 'w') as f:
        w = csv.writer(f, delimiter=delimiter)
        for key in sorted(dict_2_save):
            w.writerow([key, dict_2_save[key]])

