from utils import get_directory_content, iter_dir
import os

path = '/Users/snv23/Desktop/ИПОТЕКА'

files = {}
# get_directory_content(path, files)
# for value in files.values():
#     print(value, end='\n\n\n')
# print(files)

print(iter_dir(path))