import yaml
from networks.model import Network
import os

if __name__ == '__main__':
    config_path = "configurations"

    for filename in os.listdir(config_path)[:-1]:
        file = open(os.path.join("configurations", filename))
        param = yaml.safe_load(file)
        param["NAME"] = filename[:-5]
        network = Network(param)
        network.train()
        network.evaluate()
