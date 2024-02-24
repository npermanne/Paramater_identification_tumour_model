import yaml
from networks.model import Network


if __name__ == '__main__':
    file = open("configurations/default.yaml")
    param = yaml.safe_load(file)
    network = Network(param)
    network.train()