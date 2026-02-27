import os
import gdown
import zipfile

#Dir of datasets
os.mkdir('./datasets/')

#Download BSDS500 
URL = "https://drive.google.com/file/d/1iB2aUKTjDK0URbvUXbXBKBYAROftRKwX/view?usp=sharing"
OUTPUT_PATH = './datasets/'
gdown.download(URL, OUTPUT_PATH, fuzzy = True)

#Unzip BSDS500
path_to_zip_file = './datasets/BSDS.zip'
path_unzip_file = './datasets/'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(path_unzip_file)

#Labels
os.replace('./data/train_val_all.lst', './datasets/BSDS/train_val_all.lst')
os.replace('./data/test.lst', './datasets/BSDS/test.lst')