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
        del c
    except:
        pass
print('activation_outputs =', activation_outputs)

layer_outputs = {}
for count in range(2, 8):
    try:
        c = PlasticRecyclableClassifier('plastics', activation='elu', conv2d_layer_count=count, epochs=30)
        output = c.fit()
        layer_outputs[count] = output.history['accuracy']
        print('%d layer accuracy' % count, str(output.history['accuracy'][-1]))
        del c
    except:
        pass
print('layer_outputs =', layer_outputs)

layer_outputs = {}
for count in range(2, 8):
    try:
        c = PlasticRecyclableClassifier('plastics', activation='swish', conv2d_layer_count=count, epochs=30)
        output = c.fit()
        layer_outputs[count] = output.history['accuracy']
        print('%d layer accuracy' % count, str(output.history['accuracy'][-1]))
        del c
    except:
        pass
print('swish_layer_outputs =', layer_outputs)

layer_outputs = {}
for count in range(2, 8):
    try:
        c = PlasticRecyclableClassifier('plastics', activation='linear', conv2d_layer_count=count, epochs=30)
        output = c.fit()
        layer_outputs[count] = output.history['accuracy']
        print('%d layer accuracy' % count, str(output.history['accuracy'][-1]))
        del c
    except:
        pass
print('linear_layer_outputs =', layer_outputs)

layer_outputs = {}
for count in range(2, 8):
    try:
        c = PlasticRecyclableClassifier('plastics', activation='relu', conv2d_layer_count=count, epochs=30)
        output = c.fit()
        layer_outputs[count] = output.history['accuracy']
        print('%d layer accuracy' % count, str(output.history['accuracy'][-1]))
        del c
    except:
        pass
print('relu_layer_outputs =', layer_outputs)
