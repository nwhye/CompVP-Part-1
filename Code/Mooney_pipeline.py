import imageio.v3 as iio
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np

image = iio.imread(uri="out/original/face_0000_light_0.png")
if image.shape[-1] == 4:
    image = ski.color.rgba2rgb(image)

fig, ax = plt.subplots()
ax.imshow(image)
plt.show()

sigma = 2.0

blurred = ski.filters.gaussian(
    image, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
fig, ax = plt.subplots()
ax.imshow(blurred)
plt.show()

gray_face = ski.color.rgb2gray(blurred)

fig, ax = plt.subplots()
ax.imshow(gray_face, cmap="gray")
plt.show()

histogram, bin_edges = np.histogram(gray_face, bins=256, range=(0.0, 1.0))

fig, ax = plt.subplots()
ax.plot(bin_edges[0:-1], histogram)
ax.set_title("Grayscale Histogram")
ax.set_xlabel("grayscale value")
ax.set_ylabel("pixels")
ax.set_xlim(0, 1.0)
plt.show()


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


otsu_threshold = min(
    np.linspace(np.min(gray_face), np.max(gray_face), 256),
    key=lambda th: otsu_intraclass_variance(gray_face, th),
)
print(otsu_threshold)

t = 0.75
binary_mask = gray_face > t

fig, ax = plt.subplots()
ax.imshow(binary_mask, cmap="gray")
plt.show()

##TODO Make cycle for all images in a folder, save all mooneys in the out folder,
##TODO Decide between auto and manual threshold by testing on recognition models