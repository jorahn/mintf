import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def get_fashion_mnist():
    data = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    return x_train, x_test, y_train, y_test

def get_ds(x_train, x_test, y_train, y_test, batch_size=64):
    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds

def get_model_mlp():
    class MLPModel(Model):
        def __init__(self):
            super(MLPModel, self).__init__()
            self.flatten = Flatten()
            self.d1 = Dense(512, activation='relu')
            self.d2 = Dense(512, activation='relu')
            self.d3 = Dense(10)

        def call(self, x):
            x = self.flatten(x)
            x = self.d1(x)
            x = self.d2(x)
            return self.d3(x)

    # Create an instance of the model
    model = MLPModel()
    return model

def get_model_conv():
    class ConvModel(Model):
        def __init__(self):
            super(ConvModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    # Create an instance of the model
    model = ConvModel()
    return model

@tf.function
def train_step(model, loss_object, optimizer, images, labels, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, loss_object, optimizer, images, labels, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
def run(batch_size=64, learning_rate=1e-3, epochs=5, model=None):
    x_train, x_test, y_train, y_test = get_fashion_mnist()
    train_ds, test_ds = get_ds(x_train, x_test, y_train, y_test, batch_size)
    
    if not model:
        model = get_model_mlp()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(model, loss_object, optimizer, images, labels, train_loss, train_accuracy)

        for test_images, test_labels in test_ds:
            test_step(model, loss_object, optimizer, test_images, test_labels, test_loss, test_accuracy)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
