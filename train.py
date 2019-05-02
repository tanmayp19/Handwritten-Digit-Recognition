from ConvNeural import ConvNeuralNet
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    cnn = ConvNeuralNet()

    if cnn.isTrained():
        if input('Trained model is exist. Overwrite it? y/n ') == "n":
            sys.exit()

  
    cnn.train()
