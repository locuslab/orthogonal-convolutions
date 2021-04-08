import sys
sys.path.append('./cifar10-fast')
from core import *
import os
from torch_backend import *

try:
    batch_size = int(os.environ['batch_size'])
except:
    batch_size = 256

print('Using batch size:', batch_size)
DATA_DIR = './data'
dataset = cifar10(DATA_DIR)
t = Timer()
print('Preprocessing training data')

train_set = list(zip(transpose(pad(dataset['train']['data'], 4)) / 255.0, dataset['train']['labels']))
print(f'Finished in {t():.2} seconds')
print('Preprocessing test data')
test_set = list(zip(transpose(dataset['test']['data']) / 255.0, dataset['test']['labels']))
print(f'Finished in {t():.2} seconds')

train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])

train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2)
test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2)

for batch in train_batches:
    break
print(batch.keys())

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
