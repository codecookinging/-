import os,sys




import numpy as np
import tensorflow as tf
import datetime

from model import VggNetModel

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)