import hw1
import numpy as np
import gzip
import random
import datetime

def random_shuffle(x, y):
    x_pick = []
    y_pick = []
    num_pick = random.sample(range(0, len(y)), len(y))
    for i in num_pick:
        x_pick += [x[i]]
        y_pick += [y[i]]
    return np.array(x_pick), np.array(y_pick)

#fashion mnist dataset path      
url_train_image = 'Fashion_MNIST_data/train-images-idx3-ubyte.gz'
url_train_labels = 'Fashion_MNIST_data/train-labels-idx1-ubyte.gz'
url_test_image = 'Fashion_MNIST_data/t10k-images-idx3-ubyte.gz'
url_test_labels = 'Fashion_MNIST_data/t10k-labels-idx1-ubyte.gz'

#use gzip open .gz to get ubyte
train_image_ubyte = gzip.open(url_train_image,'r')
train_label_ubyte = gzip.open(url_train_labels,'r')
test_image_ubyte = gzip.open(url_test_image,'r')
test_label_ubyte = gzip.open(url_test_labels,'r')

# some settings
lr = 0.01
epoch = 20
batch = 20
rate = 0.8
class_num = 10
neurons = 500
image_pixels = 784
random.seed(98)

# read bytes and split training set and validating set
y = np.frombuffer(train_label_ubyte.read(), dtype=np.uint8, offset=8)
x = np.frombuffer(train_image_ubyte.read(), dtype=np.uint8, offset=16).reshape(len(y), image_pixels)/255

train_num = rate * x.shape[0]
valid_num = (1 - rate) * x.shape[0]

train_x = x[:int(train_num)]
train_y = y[:int(train_num)]

valid_x = x[int(train_num):]
valid_y = y[int(train_num):]

# training

# randomly change the order --> mini_batch
train_x, train_y = random_shuffle(train_x,train_y)
num = len(train_x)

model = hw1.HW1()
model.settings(image_pixels, class_num, [neurons, neurons], 'ReLU', lr)
        
for e in range(epoch):
    start = 0
    end = batch
    correct = 0
    loss = 0

    for b in range(int(np.ceil(num/batch))):
        c, l = model.train(train_x[start:end], train_y[start:end])
        correct += c
        loss += l
        start += batch
        end += batch
        if end > num:
            end = num
    print('Epoch', e, end='  ')
    print('train accuracy:', correct/num, 'loss:', loss/num, end='\n\n')

model.save_model(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.npy')
print('Model saved.\n')

# validating set
num = len(valid_x)
correct, loss = model.test(valid_x, valid_y)
print('valid accuracy:', correct/num, 'loss:', loss/num)

# testing set
test_y = np.frombuffer(test_label_ubyte.read(), dtype=np.uint8, offset=8)
test_x = np.frombuffer(test_image_ubyte.read(), dtype=np.uint8, offset=16).reshape(len(test_y), image_pixels)/255

# using just-trained model
num = len(test_x)
correct, loss = model.test(test_x, test_y)
print('test accuracy:', correct/num, 'loss:', loss/num)




