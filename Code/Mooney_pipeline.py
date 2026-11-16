import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np


def otsu_intraclass_variance(image, threshold):
    """
    Otsu's intra-class variance.
    """
    return np.nansum(
        [
            np.mean(cls) * np.var(image, where=cls)
            for cls in [image >= threshold, image < threshold]
        ]
    )


def preprocess(uri):
    image = iio.imread(uri=uri)
    if image.shape[-1] == 4:
        image = ski.color.rgba2rgb(image)

    # fig, ax = plt.subplots()
    # # ax.imshow(image)
    # # plt.show()

    sigma = 2.0

    blurred = ski.filters.gaussian(
        image, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    # fig, ax = plt.subplots()
    # # ax.imshow(blurred)
    # # plt.show()

    gray_face = ski.color.rgb2gray(blurred)

    # fig, ax = plt.subplots()
    # ax.imshow(gray_face, cmap="gray")
    # plt.show()

    # histogram, bin_edges = np.histogram(gray_face, bins=256, range=(0.0, 1.0))
    #
    # fig, ax = plt.subplots()
    # ax.plot(bin_edges[0:-1], histogram)
    # ax.set_title("Grayscale Histogram")
    # ax.set_xlabel("grayscale value")
    # ax.set_ylabel("pixels")
    # ax.set_xlim(0, 1.0)
    # plt.show()

    return gray_face


if __name__ == "__main__":
    input_dir = "./out/original"
    output_dir = "./out/mooney_otsu"
    os.makedirs(output_dir, exist_ok=True)
    otsus = []
    for filename in os.listdir(input_dir):
        image = preprocess(os.path.join(input_dir, filename))
        otsu_threshold = min(
            np.linspace(np.min(image), np.max(image), 256),
            key=lambda th: otsu_intraclass_variance(image, th),
        )
        t = otsu_threshold
        otsus.append(t)
        #t = 0.75
        binary_mask = image > t
        fig, ax = plt.subplots()
        ax.imshow(binary_mask, cmap="gray")
        plt.show()
        mooney_uint8 = (binary_mask * 255).astype(np.uint8)
        save_path = os.path.join(output_dir, filename)
        iio.imwrite(save_path, mooney_uint8)
    plt.hist(otsus, bins=256)
    plt.show()

##TODO Decide between auto and manual threshold by testing on recognition models