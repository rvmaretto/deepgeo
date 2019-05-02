# This file contains some functions to access filesystem
import os
import errno
from shutil import rmtree


# Create a directory if it does not exists
def mkdir(path_dir):
	try:
		os.makedirs(path_dir)
	except OSError as exc:
		if exc.errno == errno.EEXIST:
			pass
		else:
			raise


def delete_dir(path_dir):
	try:
		# os.rmdir(path_dir)
		rmtree(path_dir)
	except OSError as exc:
		if exc.errno == errno.EEXIST:
			pass
		else:
			raise