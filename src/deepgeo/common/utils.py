def check_dict_parameters(params, mandatory=[], default={}):
    for par in mandatory:
        if par not in params:
            raise AttributeError('Mandatory argument "raster_array" does not exists in parameters!')

    for par, value in default.items():
        if par not in params:
            params[par] = value

    return params
