import random

import jinja2
from itertools import product

NUMBER_OF_PARAMETERS = 6
SEED = 23

MIN_LAMBDA = 30
MAX_LAMBDA = 100

random.seed(0)

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

experiment_tuples = list(product(*parameters))
experiment_lists = [list(t) for t in experiment_tuples]

for exp in experiment_lists:
    lambda_value = random.randint(MIN_LAMBDA, MAX_LAMBDA)
    exp.append(lambda_value)


with open('queue_config.json.jinja2') as f:
    template = jinja2.Template(f.read())

    with open('example_arrival_config.json', 'w') as f2:
        f2.write(template.render(experiments=experiment_lists))

