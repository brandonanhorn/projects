import pandas as pd
import matplotlib as plt
import os

dir = 'Images/'

fruits = []
for subdir, dirs, files in os.walk(dir):
    for file in files:
        fruits.append(os.path.join(subdir,file))
