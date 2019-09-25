# This file contains some functions to access filesystem
import os
import errno
from shutil import rmtree


def mkdir(path_dir):
	""" Create a directory if it does not exists.

	This function creates a directory when it does not exists. If the directory already exists, nothing changes.

	Args:
		path_dir (str): Path to the directory to be created.
	"""
	try:
		os.makedirs(path_dir)
	except OSError as exc:
		if exc.errno == errno.EEXIST:
			pass
		else:
			raise


def delete_dir(path_dir):
	""" Deletes a directory and its contents.

	This function deletes a directory and its contents.

	Args:
		path_dir (str): Path to the directory to be deleted.
	"""
	try:
		# os.rmdir(path_dir)
		rmtree(path_dir)
	except OSError as exc:
		if exc.errno == errno.EEXIST:
			pass
		else:
			raise