import jinja2
from itertools import product

NUMBER_OF_PARAMETERS = 6
SEED = 23

cores = ["500m", "2000m"]
memories = ["1Gi", "2Gi"]
train_batches = [32, 128]
test_batches = [32, 128]
parallel_list = [2, 5]
networks = [
    f'{{ "seed": {SEED}, "network": "Cifar10CNN", "lossFunction": "CrossEntropyLoss", "dataset": "cifar10" }}',
    f'{{ "seed": {SEED}, "network": "Cifar10ResNet", "lossFunction": "CrossEntropyLoss", "dataset": "cifar10" }}'
]

parameters = [
    cores, memories, train_batches, test_batches, parallel_list, networks
]

experiments = list(product(*parameters))

with open('example_arrival_config.json.jinja2') as f:
    template = jinja2.Template(f.read())

    with open('example_arrival_config_2.json', 'w') as f2:
        f2.write(template.render(experiments=experiments))
