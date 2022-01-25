from PIL import ImageDraw, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt



def get_normalized_images():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    return (x_train, y_train), (x_test, y_test)


# function to generate noisy image. Type of noise or noises can be specified, see noise_types
def noisy_image(data, noise_function, *args):
    for ar in args:
        data += ar
    return np.clip(data + noise_function, 0, 1)


def gauss_noise(data, mean=0, var=0.2):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, data.shape)
    return gauss

def gauss_noise_preproc(image):
    return np.random.normal(0, 0.1**0.5, image.shape)

# brightness range (-1,1)
def brightness(data, value):
    return tf.image.adjust_brightness(data, value)


def contrast(data, value):
    return tf.image.adjust_contrast(data, value)


## draws 5x5 black square on image at a random position
def draw_black_square(image):
  aug_img = Image.fromarray((image*255).astype(np.uint8)) # reverse normalization since PIL library does not support floats in RGB images
  start_point = np.random.randint(0,27)
  img_draw = ImageDraw.Draw(aug_img)
  img_draw.rectangle([(start_point, start_point), (start_point+5, start_point+5)], fill="black")
  return np.array(aug_img) / 255.0  ## normalize again


def occlusion(data):
    aug_data = []
    for img in data:
      aug_data.append(draw_black_square(img))
    return np.array(aug_data)

def flip_image(image):
    # return tf.image.flip_up_down(image) both works
   return tf.image.flip_left_right(image)


def rotate(image, k):
    #  k no of times image to be rotated
    return tf.image.rot90(image, k)


# randomly change intensities of single pixels
def perturbSinglePixels(image):
    mask = np.random.randint(0, 2, size=image.shape).astype(bool)
    r = np.random.rand(*image.shape) * np.max(image)
    image[mask] = image[mask]*r[mask]
    return image


def randPerturb(image):
    rand = np.random.rand(1)
    if rand >= 0.7:
        # change pixel intensities
        return perturbSinglePixels(image)
    elif rand >= 0.4:
        # randomly change contrast of picture
        return tf.image.adjust_contrast(image,np.random.uniform(0.3,1.8))
    else:
        return image

# plot first 25 images
def plotImages(images):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()

def getImagesFlip(images):
    img_flip = images[:26]
    for i in range(25):
        img_flip[i] = tf.image.flip_left_right(images[i])
    return img_flip

# plot first 10 images and compare it with second set of images
# intended for comparing original vs decoded representation
def plotAndCompareOrigDeco(images1, images2):
    rows = 2
    cols = 10
    plt.figure(figsize=(10,10))
    for i in range(10):
        plt.subplot(rows, cols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images1[i], cmap=plt.cm.binary)
        plt.subplot(rows, cols, i+11)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images2[i], cmap=plt.cm.binary)
    plt.show()


# plot first 10 images and compare it with two other sets of images
# intended for comparing original noisy and decoded images
def plotAndCompareImages(images_clean, images_noisy, images_recon):
    fig = plt.figure(constrained_layout=True,figsize=(14, 10))
    images_list = [images_clean, images_noisy, images_recon]
    titles = ["clean images", "noisy images", "reconstructed images"]

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(titles[row])

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=10)
        for col, ax in enumerate(axs):
            ax.grid(False)
            ax.imshow(images_list[row][col], cmap=plt.cm.binary)
    plt.show()


def plotValLossAndLoss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.ylim([0, max([max(loss), max(val_loss)])])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)


# plotting different noisy pictures
def main1():
    # load the data
    (x_train, y_train), (x_test, y_test) = get_normalized_images()

    data = x_test[0:25].copy()
    plotImages(data)

    # gauss noise
    noisy_gauss = noisy_image(data, gauss_noise(data))
    plotImages(noisy_gauss)
    #
    # # brightness change
    brightness_change = noisy_image(data, brightness(data,0.1))
    plotImages(brightness_change)

    # flip change
    flip_change = flip_image(data)
    plotImages(flip_change)

    # # rotation change
    rotate_change = rotate(data, 2)
    plotImages(rotate_change)
    #
    # # brightness change
    brightness_change = noisy_image(data, brightness(data,0.1))
    plotImages(brightness_change)
    occlusion_change = occlusion(data)
    plotImages(occlusion_change)
    #
    perturbations = []
    for img in data:
        perturbations.append(randPerturb(img))
    plotImages(np.array(perturbations))


if __name__ == "__main__":
    main1()
