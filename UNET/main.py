from DataLoader import create_dataset
from ModelBuilder import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
from config import OUTPUT, N_EPOCH, VALIDATION_SPLIT, BATCH_SIZE


def train_model(start_neurons, conv_kernel_size, activation_type,
                learning_rate):
    X, Y = create_dataset()
    model = build_model(start_neurons, conv_kernel_size, activation_type,
                        learning_rate)

    callbacks = [
        ModelCheckpoint(f'{OUTPUT}model_UNET.h5',
                        save_weights_only=True,
                        verbose=1),
        # CSVLogger(f'{OUTPUT}logs.csv', separator=',',
        #           append=False)
    ]

    model.fit(X, Y, validation_split=VALIDATION_SPLIT,
              epochs=N_EPOCH, verbose=1, batch_size=BATCH_SIZE,
              callbacks=callbacks)


if __name__ == '__main__':
    train_model(16, (3, 3), 'relu', 1e-4)
