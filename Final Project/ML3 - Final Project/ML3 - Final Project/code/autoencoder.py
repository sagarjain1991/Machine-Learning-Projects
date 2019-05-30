import numpy as np
np.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
	
def classify(x):
    return np.where(x > 0.5, 1, 0)

class AutoEncoder(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if rng is None:
            rng = np.random.RandomState(1234)
            
        if W is None:
            a = 1. / n_visible
            W = np.array(rng.uniform( low=-a, high=a, size=(n_visible, n_hidden)))

        if hbias is None:
            hbias = np.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = np.zeros(n_visible)  # initialize v bias 0

        self.rng = rng
        self.x = input
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias

        
    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        return self.rng.binomial(size=input.shape, n=1, p=1-corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(np.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(np.dot(hidden, self.W_prime) + self.vbias)


    def train(self, lr=0.1, corruption_level=0.3, input=None):
        if input is not None:
            self.x = input

        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W =  np.dot(tilde_x.T, L_h1) + np.dot(L_h2.T, y)

        self.W += lr * L_W
        self.hbias += lr * np.mean(L_hbias, axis=0)
        self.vbias += lr * np.mean(L_vbias, axis=0)

		
    def negative_log_likelihood(self, corruption_level=0.3):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        epsilon = 0.00001
        z += epsilon
        print(z)
        cross_entropy = - np.mean(np.sum(self.x * np.log(z) + (1 - self.x) * np.log(1 - z), axis=1))
        return cross_entropy

    def mse(self, corruption_level=0.3):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        return np.mean((z - self.x)**2)


    def predict(self, x, type='classify'):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        if type == 'classify':
            print('classify')
            return classify(z)
        elif type == 'regression':
            print('regression')
            return z


def test_dA(learning_rate=0.1, corruption_level=0.3, training_epochs=50):
    data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    rng = np.random.RandomState(123)

    # construct dA
    da = AutoEncoder(input=data, n_visible=20, n_hidden=5, rng=rng)

    print('Training Auto Encoders:\n')
    # train
    for epoch in range(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level)
        #cost = da.negative_log_likelihood(corruption_level=corruption_level)
        cost = da.mse(corruption_level=corruption_level)
        if epoch % 5 == 0:
            print('Training epoch %d, cost is ' % epoch, cost)
        learning_rate *= 0.95


    # test
    x = np.array([[1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]])

    print('\n\nPrediction on test data: \n')
    print(da.predict(x))



if __name__ == "__main__":
    test_dA()
