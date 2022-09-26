import random
import time

from tqdm import tqdm

p_bar = tqdm(range(10), desc="A Processing Bar Sample: ", total=10, ncols=100)

for i in p_bar:
    time.sleep(random.random())

p_bar.close()