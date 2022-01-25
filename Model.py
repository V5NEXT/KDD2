import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Data_Generator import *
from Model_Architectures import *
from numpy.random import seed
import random
import os


## alternative loss function for image comparison
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


def main():
    '''     ########## TRAIN OR CHOOSE SAVED MODEL ##########       '''
    # if you want to use a already trained (and saved) model set True
    use_saved_model = False
    load_model_name = "model7_jh"  # model to be loaded
    # use different names to save trained models
    save_model_name = "new_model"

    current_dir = os.getcwd()
    filepath = os.path.join(current_dir, "models/" + save_model_name)

    # set seed for reproducibility
    seed(42)
    tf.random.set_seed(42)

    # get normalized data
    (x_train, y_train), (x_test, y_test) = get_normalized_images()

    # generate noisy data
    x_train_bright = tf.image.adjust_brightness(x_train, 0.3)
    x_train_noisy = noisy_image(x_train, gauss_noise(x_train, var=0.08))
    x_train_flipped = flip_image(x_train)
    # x_train_flipped = tf.image.flip_left_right(x_train)
    x_train_occlusion = occlusion(x_train)
    x_test_bright = np.clip(tf.image.adjust_brightness(x_test, 0.4),0,1)
    x_test_noisy = noisy_image(x_test, gauss_noise(x_test, var=0.08))
    x_test_flipped = flip_image(x_test)
    # x_test_flipped = tf.image.flip_left_right(x_test)
    x_test_occlusion = occlusion(x_test)


    # concat augmentations
    # TODO reduce size of data somehow since 5 augmentations would be 250.000 images
    x_train_augmented = np.concatenate([x_train_noisy, x_train_occlusion, x_train_flipped])
    x_test_augmented = np.concatenate([x_test_noisy, x_test_occlusion, x_test_flipped])
    x_train = np.concatenate([x_train, x_train])
    x_test = np.concatenate([x_test, x_test])
    '''     #########################################'''
    '''     ########## MODEL CONFIGURATION ##########'''
    '''     #########################################'''
    # model callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=0)  # val_loss
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="loss", save_best_only=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True)

    callbacks = [early_stopping, model_checkpoint]

    # load the model
    model = get_model4(x_train[0].shape)
    model.summary()

    # #print dimensions of layers
    # for layer in model.layers:
    #    print(layer.output_shape)

    '''  model training settings    '''
    epochs = 10  # 30
    batch_size = 128
    lr = 1e-3
    loss = ["mse", ssim_loss]  # no categorical loss since we want to reconstruct and not classify
    #    sampler = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
    #                                                              samplewise_center=False,
    #                                                              brightness_range=(0.3, 1.7),
    #                                                              samplewise_std_normalization=False,
    #                                                              preprocessing_function=draw_black_square,
    #                                                              validation_split=0.2
    #                                                              )#.flow(x=x_train, y=None, batch_size=batch_size, subset='training')

    if use_saved_model:
        filepath = os.path.join(current_dir, "models/" + load_model_name)
        model = tf.keras.models.load_model(filepath)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss[0])
        # # train model with already generated data
        history = model.fit(x=x_train_augmented, y=x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                            validation_split=0.2, callbacks=callbacks)

        # train model with imageDataGenerator
        #        sampler.fit(x_train)
        #        history = model.fit(sampler.flow(x=x_train, y=x_train, batch_size=batch_size, subset='training'),
        #                            epochs=epochs, callbacks=callbacks,
        #                            validation_data=sampler.flow(x=x_train, y=x_train, batch_size=batch_size, subset='validation'))
        # plot training history
        plotValLossAndLoss(history)

    # clean/noisy/decode test images and compare them with original
    # shuffle arrays (for plot comparison)
    p = np.random.permutation(x_test.shape[0])
    x_test = x_test[p]
    x_test_augmented = x_test_augmented[p]

    plotImages(x_test_augmented)
    test_imgs = model.predict(x_test_augmented)
    plotAndCompareImages(x_test, x_test_augmented, test_imgs)
    # evaluate loss for model comparison
    score = model.evaluate(x=x_test_augmented, y=x_test)
    print(f"Loss: {score}")
    return 0


if __name__ == "__main__":
    main()

    # loss starts at like 0.03
    # after first epoch   0.01