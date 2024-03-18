import yaml
from networks.model import Network

if __name__ == '__main__':
    file = open("configurations/cell_cycle_study.yaml")
    param = yaml.safe_load(file)
    network = Network(param)
    network.train()
    # network.load_weights()
    network.evaluate()
