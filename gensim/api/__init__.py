import subprocess
import json
import sys
import os
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
def download(file):
	url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/download/"
	data = catalogue()
	corpuses = data['gensim']['corpus']
	models = data['gensim']['model']
	if file not in corpuses and file not in models: 
		print("Incorrect corpus/model name.")
	else:
		url = url+file+"/"+file+".tar.gz"
		print("Downloading {m}".format(m=file))
		subprocess.call([sys.executable, '-m','pip','install','--no-cache-dir',url],env = os.environ.copy())

def catalogue(print_list = 0):
	url  = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/list.json"
	response = urlopen(url)
	data = response.read().decode("utf-8")
	if print_list == 1 :
		print(json.loads(data))
	else:
		return json.loads(data)
	
