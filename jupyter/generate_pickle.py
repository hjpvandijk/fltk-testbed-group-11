import os
import pickle
import pandas as pd

from tflogs2pandas import main

for root, dirs, files in os.walk("./logging/0"):
    experiment_dict = {}

    for file in files:
        if file.startswith("events.out.tfevents"):
            main(root, False, True, out_dir=f'csvs/{file}')

            df = pd.read_csv(f'csvs/{file}/all_training_logs_in_one_file.csv')

            experiment_dict[file] = df.pivot_table(index=['step'], columns=['metric'], values='value')

    with open('all_experiments.pickle', 'wb') as handle:
        pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
