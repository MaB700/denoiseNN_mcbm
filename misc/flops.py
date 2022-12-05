import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input

from keras_flops import get_flops


def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(72, 32, 1)))
    model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(1, kernel_size=3, activation='tanh', padding='same'))
    model.summary()
    model.save("model.h5")

""" def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()


    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops """

""" 
flops = get_flops("./model.h5")
print(flops) """
""" 
inp = Input((16, 16, 3))
x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(10, activation="softmax")(x)
model = Model(inp, out) """
create_model()
model = tf.keras.models.load_model("./model.h5")
# Calculae FLOPS
flops = get_flops(model, batch_size=1)
print(flops)
print(f"FLOPS: {flops / 10 ** 9:.03} G")