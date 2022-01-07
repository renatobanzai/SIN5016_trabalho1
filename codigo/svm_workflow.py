from sin5016 import svm_gpu
from sin5016 import dataprep
import sys
import logging
logging.basicConfig(filename="svm_workflow.py.log", format='%(asctime)s %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import numpy as np
import cupy as cp
import math
from cvxopt import matrix, solvers
import matplotlib.pylab as plt
import h5py
import datetime
import pickle
import random

dataprep = dataprep.dataprep("/home/madeleine/Documents/mestrado/5016/trabalho/data/hog_11_15_20_56")
dataprep.get_dictionary_artists()
print()