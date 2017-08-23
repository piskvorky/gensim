from __future__ import print_function
from __future__ import absolute_import
import json
import os
import tarfile
import logging
import sys
import shutil
import errno
try:
	import urllib.request as urllib
except ImportError:
	import urllib

try:
	from urllib.request import urlopen
except ImportError:
	from urllib2 import urlopen


user_dir = os.path.expanduser('~')
base_dir = os.path.join(user_dir, 'gensim-data')
log_file_dir = os.path.join(base_dir, 'api.log')
if not os.path.isdir(base_dir):
	try:
		os.makedirs(base_dir)
	except OSError as e:
		if e.errno == errno.EEXIST:
			raise Exception(
				"Not able to create folder gensim-data in {}. File gensim-data "
				"exists in the direcory already.".format(user_dir))
		else:
			raise Exception(
				"Can't create {}. Make sure you have the read/write permissions "
				"to the directory or you can try creating the folder manually"
				.format(base_dir))

logging.basicConfig(
	format='%(asctime)s :%(name)s :%(levelname)s : %(message)s',
	filename=log_file_dir, level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger = logging.getLogger('gensim.api')
logger.addHandler(console)


def download(file_name):
	url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/download/{f}/{f}.tar.gz".format(f=file_name)
	data_folder_dir = os.path.join(base_dir, file_name)
	data = catalogue(print_list=False)
	corpuses = data['gensim']['corpus']
	models = data['gensim']['model']
	if file_name not in corpuses and file_name not in models:
		logger.error(
			"Incorect Model/corpus name. Use catalogue(print_list=TRUE) or"
			" python -m gensim -c to get a list of models/corpuses"
			" available.")
		sys.exit(0)
	compressed_folder_name = "{f}.tar.gz".format(f=file_name)
	compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
	is_installed = False
	is_downloaded = False
	installed_message = "{f} installed".format(f=file_name)
	downloaded_message = "{f} downloaded".format(f=file_name)
	if os.path.exists(data_folder_dir):
		log_file_dir = os.path.join(base_dir, 'api.log')
		with open(log_file_dir) as f:
			f = f.readlines()
		for line in f:
			if installed_message in line:
				print("{} has already been installed".format(file_name))
				is_installed = True
				sys.exit(0)
	if os.path.exists(data_folder_dir) and not is_installed:
		shutil.rmtree(data_folder_dir)
		for line in f:
			if downloaded_message in line:
				is_downloaded = True
				break
	if not is_downloaded:
		os.makedirs(data_folder_dir)
		logger.info("Downloading %s", file_name)
		urllib.urlretrieve(url, compressed_folder_dir)
		data_url = data_links(file_name)
		if data_url is not None:
			index = data_url.rfind("/")
			data_dir = os.path.join(data_folder_dir, data_url[index+1:])
			urllib.urlretrieve(data_url, data_dir)
		logger.info("%s downloaded", file_name)
	if not is_installed:
		logger.info("Creating %s", data_folder_dir)
		if os.path.exists(data_folder_dir):
			logger.info("Creation of %s successful.", data_folder_dir)
			tar = tarfile.open(compressed_folder_dir)
			logger.info("Extracting files from %s", data_folder_dir)
			tar.extractall(data_folder_dir)
			tar.close()
			logger.info("%s installed", file_name)
		else:
			logger.error(
				"Not able to create %s. Make sure you have the correct read/"
				"write permissions for %s or you can try creating it manually",
				data_folder_dir, base_dir)


def catalogue(print_list=False):
	url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/list.json"
	response = urlopen(url)
	data = response.read().decode("utf-8")
	data = json.loads(data)
	if print_list:
		corpuses = data['gensim']['corpus']
		models = data['gensim']['model']
		print("Corpuses available : ")
		for corpus in corpuses:
			print(corpus)
		print("")
		print("Models available : ")
		for model in models:
			print(model)
	return data


def info(file_name):
	data = catalogue(False)
	corpuses = data['gensim']['corpus']
	models = data['gensim']['model']
	if file_name in corpuses:
		print(data['gensim']['corpus'][file_name])
	elif file_name in models:
		print(data['gensim']['model'][file_name])
	else:
		catalogue(print_list=True)
		raise Exception(
			"Incorrect model/corpus name. Choose the model/corpus from the list "
			"above.")


def load(file_name, return_path=False):
	user_dir = os.path.expanduser('~')
	base_dir = os.path.join(user_dir, 'gensim-data')
	folder_dir = os.path.join(base_dir, file_name)
	if not os.path.exists(folder_dir):
		raise Exception(
			"Incorrect model/corpus name. Use catalogue(print_list=True) to get a list of "
			"avalible models/corpus. If the model/corpus name you entered is"
			" in the catalogue, then please download the model/corpus by "
			"calling download('{f}') function".format(f=file_name))
	elif return_path:
		return folder_dir
	else:
		sys.path.insert(0, base_dir)
		module = __import__(file_name)
		data = module.load_data()
		return data


def data_links(file_name):
	url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/links.json"
	response = urlopen(url)
	data = response.read().decode("utf-8")
	data = json.loads(data)
	if file_name in data['data_links']:
		return data['data_links'][file_name]['link']
