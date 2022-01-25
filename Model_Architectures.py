import tensorflow as tf
from tensorflow.keras import regularizers

def get_model(shape):
    # TODO use regularization (kernel_regularizer, bias_regularizer, dropout etc...)
    # TODO improve accuracy (aka encoding/decoding) of model (check with plotAndComapreImages())
    # TODO try out different approaches

    # define encoder
    input_img = tf.keras.Input(shape=(shape[0], shape[1], shape[2]))

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same")(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    encoded = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same")(x)
    encoded = tf.keras.layers.BatchNormalization()(encoded)

    # define decoder
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same")(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    decoded = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation="sigmoid", padding="same")(x)

    # final model
    model = tf.keras.Model(input_img, decoded)
    return model

# Settings:
#    epochs = 10, batch_size = 128, lr = 1e-3, loss = "mse"
# Results:
# good performance on gaussian noise:            Loss: 0.008053826168179512
# moderate performance on horizontal flip:       Loss: 0.014129387214779854
def get_model1(shape):
    # define encoder
    input_img = tf.keras.Input(shape=(shape[0], shape[1], shape[2]))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    encoded = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=1)(x)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.ReLU()(encoded)

    # define decoder

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    decoded = tf.keras.layers.Conv2D(filters=shape[2], kernel_size=(3,3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(input_img, decoded)
    return model


def get_model2(shape):
    # TODO use regularization (kernel_regularizer, bias_regularizer, dropout etc...)

    # define encoder
    input_img = tf.keras.Input(shape=(shape[0], shape[1], shape[2]))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    encoded = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=1)(x)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.ReLU()(encoded)

    # define decoder

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    decoded = tf.keras.layers.Conv2D(filters=shape[2], kernel_size=(3,3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(input_img, decoded)
    return model

# Settings:
#    epochs = 10, batch_size = 128, lr = 1e-3, loss = "mse"
'''
Results:
On Gauss noise:
First try: too much regularization -> loss: 0.0327
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)
Second: much better but looks blurry, Loss: 0.006185444537550211 (see Model3_rescon_gauss_with_L1L2_reg.png)
    kernel_regularizer=regularizers.l1_l2(l1=1e-7, l2=1e-6),bias_regularizer=regularizers.l2(1e-6),activity_regularizer=regularizers.l2(1e-7)
    On flips: Loss: 0.015383195132017136, (see: Model3_recon_flips_with_L1L2_reg.png)
Third: on flips: kernel_regularizer=regularizers.l1(l1=1e-7)
    Loss: 0.014915083535015583
    on gauss: Loss: 0.0068697091192007065
Fourth: on flips: kernel_regularizer=regularizers.l1(l1=1e-9)
    Loss: 0.014711891300976276
'''
def get_model3(shape):
    # define encoder
    input_img = tf.keras.Input(shape=(shape[0], shape[1], shape[2]))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same",kernel_regularizer=regularizers.l1(l1=1e-7))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    encoded = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=1)(x)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.ReLU()(encoded)

    # define decoder

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same",kernel_regularizer=regularizers.l1(l1=1e-7))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    decoded = tf.keras.layers.Conv2D(filters=shape[2], kernel_size=(3,3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(input_img, decoded)
    return model


'''
Results:

model7_jh (on occlusion): loss: 0.00164
model8_jh (on noise): (lr:3e-3) loss: 0.0060
model8_jh (on noise); (lr:1e-3) loss: 0.0058
model9_jh (occlusion + noise): 0.0036
'''
def get_model4(shape):
    # define encoder
    input_img = tf.keras.Input(shape=(shape[0], shape[1], shape[2]))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", kernel_regularizer=regularizers.l1(l1=1e-7))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    encoded = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=1)(x)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.ReLU()(encoded)

    # define decoder

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    decoded = tf.keras.layers.Conv2D(filters=shape[2], kernel_size=(3,3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(input_img, decoded)
    return model
