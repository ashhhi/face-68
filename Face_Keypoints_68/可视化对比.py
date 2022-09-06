import os
import pandas as pd
from matplotlib import pyplot as plt

root_path = 'log/'
dirs = os.listdir(root_path)

for dir in dirs:
    path = root_path + dir
    data = pd.read_csv(path)
    print(dir)
    plt.plot(data['train Loss'],label=dir.split('.')[0])
#
for dir in dirs:
    path = root_path + dir
    data = pd.read_csv(path)
    print(dir)
    plt.plot(data['val loss'],label=dir.split('.')[0],linestyle=':')
plt.title('loss')
plt.legend()
plt.savefig('log/loss.jpg')
plt.show()



for dir in dirs:
    path = root_path + dir
    data = pd.read_csv(path)
    print(dir)
    plt.plot(data['training accuracy'],label=dir.split('.')[0])

for dir in dirs:
    path = root_path + dir
    data = pd.read_csv(path)
    print(dir)
    plt.plot(data['val accuracy'],label=dir.split('.')[0],linestyle=':')
plt.title('accuracy')
plt.legend()
plt.savefig('log/accuracy.jpg')
plt.show()

