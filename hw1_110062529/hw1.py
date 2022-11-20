import numpy as np

##START YOUR CODE
class HW1():
    def __init__(self, ):
        self._category = 0
        self._layer = 0
        self._neuron = []
        self._activation_type = ''
        self._lr = 0
        self._w = []
        self._b = []
        self._d_input = np.array([])

    def settings(self, x, category, neural, acti, lr):
        self._activation_type = acti
        self._lr = lr
        self._category = category
        self._neural = neural
        self._neural.extend([self._category, x])
        self._layer = len(self._neural)
        
        # initialize weight and bias with sqrt(2/neural)
        for l in range(self._layer-1):
            self._w.append(np.random.randn(self._neural[l],self._neural[l-1]) * np.sqrt(2. / self._neural[l]))
            self._b.append(np.random.randn(self._neural[l])* np.sqrt(2. / self._neural[l]))

    def _one_hot_encode(self, y):
        one_hot = np.zeros(self._category)
        one_hot[y] = 1
        return one_hot

    def _activation(self, val):
        if self._activation_type == 'sigmoid':
            return 1/(1+np.exp(-val))
        elif self._activation_type == 'ReLU':
            return np.where(val > 0, val, 0)
        elif self._activation_type == 'tanh':
            return (np.exp(2 * val) - 1)/(np.exp(2 * val) + 1)
        
    def _activation_d(self, val):
        if self._activation_type == 'sigmoid':
            return val*(1-val)
        elif self._activation_type == 'ReLU':
            return np.where(val>0, 1, 0)
        elif self._activation_type == 'tanh':
            t = (np.exp(2 * val) - 1)/(np.exp(2 * val) + 1)
            return 1 - t**2
        
    def _softmax(self, out):
        return np.exp(out-np.max(out))/np.sum(np.exp(out-np.max(out)))

    def _cross_entropy_loss(self, y, prob):
        epsilon = 0.000001
        return np.max(-y*np.log(prob + epsilon))
    
    def _forward_pass(self, x):
        out = [x]
        z = np.array([])
        for l in range(self._layer-1):
            z = self._w[l].dot(out[l])+self._b[l]
            if l != self._layer-2:
                out.append(self._activation(z))
        prob = self._softmax(z)
        return out, prob
    
    def _back_pass(self, y, prob, out):
        d_w_all = []
        d_b_all = []

        # combine dloos/dp and dp/dna, to prevent dloss/dp becomig Inf when p is zero.
        # doing back pass with chain rule
        d_na = prob - y 
        
        # start from the output layer
        for l in range(self._layer-2, -1, -1):
            d_w = d_na[np.newaxis].transpose().dot(out[l][np.newaxis])

            d_b = d_na
            d_w_all.append(d_w)
            d_b_all.append(d_b)

            d_a = self._w[l].transpose().dot(d_na)
            d_na = self._activation_d(out[l]) * d_a

        d_w_all.reverse()
        d_b_all.reverse()
        self._d_input = d_a
        return d_w_all, d_b_all

    def train(self, data, y):
        batch = len(data)
        loss = 0
        correct = 0
        d_w_batch = []
        d_b_batch = []

        for l in range(self._layer-1):
            d_w_batch.append(np.zeros([self._neural[l], self._neural[l-1]]))
            d_b_batch.append(np.zeros(self._neural[l]))

        # calculate each data's gradient of the batch
        for num in range(batch):
            a, prob = self._forward_pass(data[num])
            one_hot_label = self._one_hot_encode(y[num])
            temp_d_w, temp_d_b = self._back_pass(one_hot_label, prob, a)
            
            # calculate loss
            if np.array_equal(one_hot_label, prob//prob.max()):
                correct += 1    
            loss += self._cross_entropy_loss(one_hot_label, prob)
            
            # sum the gradient
            for l in range(self._layer-1):
                d_w_batch[l] = d_w_batch[l]+temp_d_w[l]
                d_b_batch[l] = d_b_batch[l]+temp_d_b[l]
                
        # take the average gradient to update weight and bias
        for l in range(self._layer-1):
            self._w[l] = self._w[l]-d_w_batch[l]/batch*self._lr
            self._b[l] = self._b[l]-d_b_batch[l]/batch*self._lr
        return correct, loss

    def test(self, data, y):
        batch = len(data)
        loss = 0
        correct = 0

        for num in range(batch):
            _, prob = self._forward_pass(data[num])
            one_hot_label = self._one_hot_encode(y[num])
            if np.array_equal(one_hot_label, prob//prob.max()):
                correct += 1
            loss += self._cross_entropy_loss(one_hot_label, prob)
        return correct, loss

    def predict(self, x):
        _, prob = self._forward_pass(x)
        return int(np.argmax(prob))

    def get_input_gradient(self):
        return self._d_input

    def save_model(self, filename):
        with open(filename, 'w') as f:
            
            # write neural, activation type, and learning rate
            f.write(','.join(map(str, self._neural)))
            f.write('\n')
            f.write(self._activation_type)
            f.write('\n')
            f.write(str(self._lr))
            f.write('\n')

            # write weight and bias
            for l in range(self._layer-1):
                for n in range(self._neural[l]):
                    f.write(','.join(map(str, self._w[l][n])))
                    f.write('\n')
            for l in range(self._layer-1):
                f.write(','.join(map(str, self._b[l])))
                f.write('\n')
            f.write(','.join(map(str, self._d_input)))
            f.write('\n')

    def load_model(self, filename):
        with open(filename, 'r') as f:
            # read info to reconstruct the network
            temp_neural = f.readline().strip('\n').split(',')
            self._neural = list(map(int, temp_neural))
            self._layer = len(self._neural)
            self._category = self._neural[-2]
            self._activation_type = f.readline().strip('\n')
            self._lr = float(f.readline().strip('\n'))
            self._w = []
            self._b = []
            
            # read weight and bias for each layer
            for l in range(self._layer-1):
                layer_w = []
                for n in range(self._neural[l]):
                    temp_w = f.readline().strip('\n').split(',')
                    layer_w.append(list(map(float, temp_w)))
                self._w.append(np.array(layer_w))
            for l in range(self._layer-1):
                temp_b = f.readline().strip('\n').split(',')
                self._b.append(np.array(list(map(float, temp_b))))
            temp_a = f.readline().strip('\n').split(',')
            self._d_input = np.array(temp_a)
    
if __name__ == '__main__':
    pass
