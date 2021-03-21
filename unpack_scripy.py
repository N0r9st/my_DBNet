import zipfile
import re
from glob import glob

datapath = 'data/zipdata/ch4_training_images.zip'
with zipfile.ZipFile(datapath, 'r') as zip_ref:
    zip_ref.extractall('data/streets/train/img')


datapath = 'data/zipdata/ch4_test_images.zip'
with zipfile.ZipFile(datapath, 'r') as zip_ref:
    zip_ref.extractall('data/streets/test/img')

datapath = 'data/zipdata/ch4_training_localization_transcription_gt.zip'
with zipfile.ZipFile(datapath, 'r') as zip_ref:
    zip_ref.extractall('data/streets/train/gt')


datapath = 'data/zipdata/Challenge4_Test_Task1_GT.zip'
with zipfile.ZipFile(datapath, 'r') as zip_ref:
    zip_ref.extractall('data/streets/test/gt')


gt = sorted(glob('data/streets/train/gt/*'))
img = sorted(glob('data/streets/train/img/*'))


with open('data/streets/train.txt', 'w') as f:
    for img1, gt1 in zip(img,gt):
        f.write(img1+'\t'+gt1+'\n')


gt = sorted(glob('data/streets/test/gt/*'))
img = sorted(glob('data/streets/test/img/*'))

with open('data/streets/test.txt', 'w') as f:
    for img1, gt1 in zip(img,gt):
        f.write(img1+'\t'+gt1+'\n')