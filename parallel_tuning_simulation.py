from multiprocessing import Pool
import random
import time

def simulate_training(config):
    lr, batch = config
    acc = 0.8 + random.uniform(-0.05, 0.05)
    time.sleep(random.uniform(1, 2))
    return f"Config: LR={lr}, Batch={batch} => Accuracy={acc:.4f}"

configs = [(0.001, 32), (0.001, 64), (0.0001, 32), (0.0001, 64)]

if __name__ == '__main__':
    with Pool(4) as pool:
        results = pool.map(simulate_training, configs)
    for res in results:
        print(res)
