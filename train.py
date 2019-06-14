import matplotlib.pyplot as plt
from pathlib import Path

from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

from models import pelee_net


def prepare_dataset(num_classes=10):
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test


def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig("history_accuracy.png")

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig("history_loss.png")


if __name__ == '__main__':

    num_classes = 10
    batch_size = 32
    epochs = 100
    x_train, x_test, y_train, y_test = prepare_dataset(num_classes)

    log_path = Path("logs")
    if not log_path.exists():
        log_path.mkdir()

    model = pelee_net()
    model.summary()

    checkpoint_name = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(str(log_path.joinpath(checkpoint_name)),
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=200,
                               verbose=1, mode='auto')

    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)
    plot_history(history)
