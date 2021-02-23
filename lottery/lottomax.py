import random
import numpy as np

numbers = []

for i in range(1, 51):
    numbers.append(i)

winning_numbers = np.random.choice(numbers, 7, replace=False)

winning_numbers.sort()
winning_numbers
