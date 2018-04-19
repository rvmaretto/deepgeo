# This file contains some functions to access filesystem
import os
import errno
#import h5py

# Create a directory if it does not exists
def mkdir(path_dir):
	try:
		os.makedirs(path_dir)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST:
			pass
		else: raise

def delete_dir(path_dir):
	try:
		os.rmdir(path_dir)
	except OSError as exc:
		if(exc.errno == errno.EEXIST):
			pass
		else: raise

# def load_hdf5(path_file):
# 	print("\n-- Loading HDF5 file --")
# 	print("   Loading file ", path_file)

# 	with h5py.File(path_file, "r") as f:
# 		def _load_hdf5(root_data):
# 			_data = {}
# 			for key, val in root_data.items():
# 				if isinstance(val, h5py.Group):
# 					_data[key] = _load_hdf5(val)
# 				else:
# 					_data[key] = val.value
# 			return _data
# 		data = _load_hdf5(f)
# 		print("   Data succesfully loaded!")
# 		return data

# def dict_to_hdf(fname, d):
#     """
#     Save a dict-of-dict datastructure where values are numpy arrays
#     to a .hdf5 file
#     """
#     with h5py.File(fname, 'w') as f:
#         def _dict_to_group(root, d):
#             for key, val in d.items():
#                 if isinstance(val, dict):
#                     grp = root.create_group(key)
#                     _dict_to_group(grp, val)
#                 else:
#                     root.create_dataset(key, data=val)

#         _dict_to_group(f, d)