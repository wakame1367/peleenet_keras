import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                               TensorBoard)

from models import pelee_net


def get_arguments():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--epochs", default=100, type=int)
    _parser.add_argument("--batch_size", default=32, type=int)
    _args = _parser.parse_args()
    return _args


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
    args = get_arguments()
    num_classes = 10
    input_shapes = (3, 32, 32)
    x_train, x_test, y_train, y_test = prepare_dataset(num_classes)

    log_path = Path("logs")
    if not log_path.exists():
        log_path.mkdir()

    model = pelee_net(input_shapes=input_shapes)
    model.summary()

    checkpoint_name = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(str(log_path.joinpath(checkpoint_name)),
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=200,
                               verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir=str(log_path))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        callbacks=[checkpoint, early_stop, tensorboard])
    plot_history(history)
