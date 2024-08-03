import yaml
from networks.model import Network
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, default='cell_cycle_study')
parser.add_argument('-train', action='store_true')

if __name__ == '__main__':

    parser = parser.parse_args()
    file = open(os.path.join("configurations", f"{parser.config}.yaml"))
    param = yaml.safe_load(file)
    param["NAME"] = parser.config

    param["TRAINING"]["EPOCH"] = 50
    for worker in range(1, 13):
        param["TRAINING"]["NUM_WORKERS"] = worker
        start = time.time()
        network = Network(param)
        if parser.train:
            network.train()
        else:
            network.load_weights()
        network.evaluate()
        end = time.time()
        print(f"Elapsed time: {end - start} for {worker} worker(s)")
