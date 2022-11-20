import hw1
import numpy as np
import gzip


url_test_image = 'Fashion_MNIST_data/t10k-images-idx3-ubyte.gz'
url_test_labels = 'Fashion_MNIST_data/t10k-labels-idx1-ubyte.gz'
test_image_ubyte = gzip.open(url_test_image,'r')
test_label_ubyte = gzip.open(url_test_labels,'r')

image_pixels = 784

test_y = np.frombuffer(test_label_ubyte.read(), dtype=np.uint8, offset=8)
test_x = np.frombuffer(test_image_ubyte.read(), dtype=np.uint8, offset=16).reshape(len(test_y), image_pixels)/255


# testing set
# 2022_10_31_18_45_44.npy can reach 87.42% accuracy
# 2022_11_01_17_26_35.npy can reach 88.17% accuracy

model = hw1.HW1()
# model.load_model('2022_10_31_18_45_44.npy')
model.load_model('2022_11_01_17_26_35.npy')
num = len(test_x)
correct, loss = model.test(test_x, test_y)
print('test accuracy:', correct/num, 'loss:', loss/num)