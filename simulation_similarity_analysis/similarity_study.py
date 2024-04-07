import matplotlib.pyplot as plt
from enum import Enum
from multiprocessing import Pool
from multiprocessing import set_start_method
import time

from utils import *





class SimilarityStudy:


    def __init__(self,*datasets_name,jupyter=False):
        self.comparator = Comparator(*datasets_name, jupyter=jupyter)




if __name__ == '__main__':
    s = SimilarityStudy("same_value_study_start=350_interval=100_ndraw=8_size=(64,64)",
                        "cell_cycle_study_start=350_interval=100_ndraw=8_size=(64,64)")


    test = s.mean_difference(SimilarityStudy.Metric.IMAGE_ABSOLUTE_DIFFERENCE, 350, "cells_types", "cell_cycle", 15,
                                      0, 12, 1000, crop_percentage=0.5)
    print(np.max(test))