import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

weight_decay = 1e-4

x = x_input = keras.layers.Input([32,32,3], name='input')
x = keras.layers.Conv2D(
    10, 3, (1,1), 'same',
    use_bias=False,
    kernel_initializer=keras.initializers.he_normal(),
    kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu', name='relu')(x)
#x = keras.layers.ReLU()(x)

x = keras.layers.GlobalMaxPool2D()(x)
x = keras.layers.Dense(
    5,
    kernel_initializer=keras.initializers.he_uniform(),
    kernel_regularizer=keras.regularizers.l2(weight_decay),
    name='output')(x)


m = keras.Model(x_input, x)

optimizer = keras.optimizers.SGD(0.1, 0.9)
m.compile(optimizer, 'mse')

input = tf.random_normal([1,32,32,3])
output = m.predict(input, steps=1)

m.save('model.h5')


# m.save_weights('save/model')

# sess = K.get_session()
# with sess.graph.as_default():
#     tf.train.write_graph(sess.graph, 'save', 'model.pbtxt')

#freeze_graph --input_graph model.pbtxt --input_checkpoint=model --output_node_names output --output_graph model_frozen
