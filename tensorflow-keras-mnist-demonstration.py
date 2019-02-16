import tensorflow as tf
import time
mnist = tf.keras.datasets.mnist

# Train 60000, test 10000
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0


# Naive NN with 3 layers, SGD 0.01 lr, cross entropy loss
def test1():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=20)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# NN with 3 layers, Adam 0.01 lr, cross entropy loss
def test2():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    print(model)
    adam = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=adam,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=2)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# Compare to test1, config changed to: lr = 0.1, epochs = 5
def test3():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# Reduce layers: only one hidden layer
def test4():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# Change hidden layer to 8 nodes
def test5():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# No hidden layer
def test6():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# Try to overfit it.
def test7():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dense(10240, activation=tf.nn.relu),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


# Add regularizations.
def test8():
    start = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation=tf.nn.relu,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(10240, activation=tf.nn.relu,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd,
                  validation_split=0.1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    result = model.evaluate(x_test, y_test)
    end = time.time()
    print(result, end - start)


test1()
