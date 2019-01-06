import tensorflow as tf
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize the input value between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    # Flatten the graph and link each pixels together
    keras.layers.Flatten(input_shape=(28, 28)),
    # Full connected layers
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Full connect, with softmax spit probability (total sum to 1)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# Adam is usually better than SGD
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
