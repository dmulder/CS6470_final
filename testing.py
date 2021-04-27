#!/usr/bin/env python3
from final import PlasticRecyclableClassifier

activation_outputs = {}
# deserialize
for activation in ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'serialize', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']:
    try:
        c = PlasticRecyclableClassifier('plastics', activation=activation)
        output = c.fit()
        activation_outputs[activation] = output.history['accuracy']
        print(activation, 'accuracy', str(output.history['accuracy'][-1]))
    except:
        pass
print('activation_outputs =', activation_outputs)

layer_outputs = {}
for count in range(2, 16):
    try:
        c = PlasticRecyclableClassifier('plastics', conv2d_layer_count=count)
        output = c.fit()
        layer_outputs[count] = output.history['accuracy']
        print('%d layer accuracy' % count, str(output.history['accuracy'][-1]))
    except:
        pass
print('layer_outputs =', layer_outputs)

epochs_outputs = {}
for epochs in range(2, 50):
    try:
        c = PlasticRecyclableClassifier('plastics', epochs=epochs)
        output = c.fit()
        epochs_outputs[epochs] = output.history['accuracy']
        print('%d epochs accuracy' % epochs, str(output.history['accuracy'][-1]))
    except:
        pass
print('epochs_outputs =', epochs_outputs)
