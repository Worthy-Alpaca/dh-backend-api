import numpy as np
from pathlib import Path
from simulation.dataloader import DataLoader

class CartSetup():
  def __init__(self, data):
    components = data[1]
    feedcart = {}
    for i in components['FeedStyle'].unique():
      feedcart[i] = components.loc[components['FeedStyle'] == i]

    self.feedcart = {k: v for k, v in feedcart.items() if k == k}

  def __call__(self):
    
    cart = 0
    print(f'Setup for this product in progess: {len(self.feedcart.keys())} Carts needed')
    for key in self.feedcart:
      cart = cart +1
      time = 0
      print(f'Setting up Cart {cart} with {len(self.feedcart[key])} components')
      complexity = len(self.feedcart[key]) / 36
      for i in range(len(self.feedcart[key])):
        time = (60 + np.random.randint(0, 30) * complexity + 9.8) + time
      print(f'Needed time: {time / 60} min')
    
    return time

  def desetup(self):

    time = 0
    for i in range(self.NumComp):
      time = (48 + np.random.randint(0, 30) + 9.9) + time

    return time

if __name__ == '__main__':
  path = Path('/content/3160194')
  dataloader = DataLoader(path, separator=',')
  data = dataloader()
  CartSetup(data)()