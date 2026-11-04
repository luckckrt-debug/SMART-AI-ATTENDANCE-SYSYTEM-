from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.models import Sequential
import warnings

warnings.filterwarnings("ignore")


class DenseArchs:
    def __init__(self, classes):
        self.classes = max(2, int(classes))
        self.model = Sequential()

    def arch(self):
        self.model.add(Dense(128, input_dim=128))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(64))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(32))
        self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(Dense(16))
        self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(Dense(self.classes))
        self.model.add(Activation('softmax'))

        return self.model
