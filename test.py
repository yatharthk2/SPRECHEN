import torch 
import numpy as np
a = np.arange(12).reshape(3,2,1)
a = torch.tensor(a)
print(a)
print(torch.max(a))
print(torch.argmax(a , dim = 2))