import numpy as np 
from decision_tree import * 
from main import *

if __name__ == '__main__':
    for pruning in (False, True):
        print(f"\n===== Pruning {'on' if pruning else 'off'} =====")
        for dataset in ("clean", "noisy"):
            print(f'==={dataset.capitalize()} Dataset===')
            dataset = load_data(f'wifi_db/{dataset}_dataset.txt')
            metrics, depth = cross_validate(dataset, prune=pruning)
            # print(f"average depth is {np.average(depth)}")
            for key, value in metrics.items():
                print(f"Mean {key} is:\n {mean(value)}")