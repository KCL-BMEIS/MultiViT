import time

import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms.functional as tvt



def augment_rand_edge_crop_v0_1(randstate):
    def _inner(images):
        cropped_images = list()
        cropped_images_meta = list()
        for image in images:
            cols, rows = image.shape[1:]
            top = randstate.randint(0, rows//10)
            bottom = randstate.randint(0, rows//10)
            left = randstate.randint(0, cols//10)
            right = randstate.randint(0, cols//10)
            cropped_image = image[..., top:rows - bottom, left:cols - right]
            cropped_images.append(cropped_image)
            cropped_images_meta.append({"crop:", (top, bottom, left, right)})
        return cropped_images, cropped_images_meta
    return _inner



def rotate90():

    def _inner(image, rng):
        rotations = rng.randint(0, 4)
        rotate_image = torch.rot90(image, rotations, dims=(1, 2))
        rotate_image_meta = {"rotate90:", rotations}
        return rotate_image, rotate_image_meta

    return _inner



def rotate(max_angle):

    def _inner(image, rng):
        rotation = (rng.rand() - 0.5) * max_angle

        rotated_image = tvt.rotate(image, rotation, expand=True)
        rotate_image_meta = {"rotate": rotation}
        return rotated_image, rotate_image_meta

    return _inner



def flip(axis):

    def _inner(image, rng):
        flip_image = torch.flip(image, axis)
        flip_image_meta = {"flip": axis}
        return flip_image, flip_image_meta

    return _inner



def crop(vertical_param, horizontal_param):

    def _inner(image, rng):
        cols, rows = image.shape[1:]
        top = rng.randint(0, rows * vertical_param)
        bottom = rng.randint(0, rows * vertical_param)
        left = rng.randint(0, cols * horizontal_param)
        right = rng.randint(0, cols * horizontal_param)
        cropped_image = image[..., top:rows - bottom, left:cols - right]
        cropped_image_meta = {"crop:", (top, bottom, left, right)}
        return cropped_image, cropped_image_meta

    return _inner


def noise(mean, variance, gen):

    def _inner(image, rng):
        noise = torch.normal(mean, variance, image.shape, generator=gen).requires_grad_(False)
        noisy_image = image + noise
        return noisy_image

    return _inner



def augment_rand_edge_crop_v0_1(randstate):
    cropper = crop(0.1, 0.1)
    def _inner(images):
        t0 = time.time()
        cropped_images = list()
        cropped_images_meta = list()
        for image in images:
            image, meta = cropper(image, randstate)
            cropped_images.append(image)
            cropped_images_meta.append(meta)
        print(f"Preprocessing time: {time.time() - t0}")
        return cropped_images, cropped_images_meta
    return _inner


def no_augment():

    def _inner(images, low_res, high_res = None):

        with torch.no_grad():
            augmented_images = list()
            augmented_images_hr = None if high_res is None else list()

            for image in images:
                image = nn.functional.interpolate(
                    image.unsqueeze(0), size=low_res, mode='bilinear', align_corners=False
                ).squeeze(0)
                augmented_images.append(image)

                if high_res is not None:

                    image_hr = nn.functional.interpolate(
                        image.unsqueeze(0), size=high_res, mode='bilinear', align_corners=False
                    ).squeeze(0)
                    augmented_images_hr.append(image_hr)

            return augmented_images, augmented_images_hr, list()

    return _inner


def augument_crop_v0_1(
        rng,
        crop_p = 1.0,
        noise_p = 0.25,
):
    rng_seeds = rng.randint(0, 2**32, size=256)
    croprng = np.random.RandomState(rng_seeds[0])
    t_cropper = crop(0.1, 0.1)

    noiserng = np.random.RandomState(rng_seeds[5])
    noisegen = torch.Generator().manual_seed(int(rng_seeds[6]))
    t_gnoise = noise(0, 0.1, gen=noisegen)

    noiserng_hr = np.random.RandomState(rng_seeds[7])
    noisegen_hr = torch.Generator().manual_seed(int(rng_seeds[8]))
    t_gnoise_hr = noise(0, 0.1, gen=noisegen_hr)


    def _inner(images, low_res, high_res = None):
        with torch.no_grad():
            augmented_images_lr = list()
            augmented_images_hr = None if high_res is None else list()
            augmented_images_meta = list()

            for image in images:
                augmented_image_meta = list()

                # shape transforms - pre interpolate

                if croprng.rand() < crop_p:
                    image, meta = t_cropper(image, croprng)
                    augmented_image_meta.append(meta)

                # interpolate low_res

                image_lr = nn.functional.interpolate(
                    image.unsqueeze(0), size=low_res, mode='bilinear', align_corners=False
                ).squeeze(0)

                # pixel transforms low_res

                if noiserng.rand() < noise_p:
                    image_lr = t_gnoise(image_lr, noiserng)

                augmented_images_lr.append(image_lr)


                if high_res is not None:

                    # interpolate high_res

                    image_hr = nn.functional.interpolate(
                        image.unsqueeze(0), size=high_res, mode='bilinear', align_corners=False
                    ).squeeze(0)

                    # pixel transforms low_res

                    if noiserng_hr.rand() < noise_p:
                        image_hr = t_gnoise_hr(image_hr, noiserng)

                    augmented_images_hr.append(image_hr)


                augmented_images_meta.append(augmented_image_meta)

            return augmented_images_lr, augmented_images_hr, augmented_images_meta

    return _inner



def augment_light_v0_1(
        rng,
        rotate90_p = 0.5,
        flip_p_h = 0.5,
        flip_p_v = 0.2,
        rotate_p = 0.5,
        crop_p = 1.0,
        noise_p = 0.25,
        blur_p = 0.25,
):
    rng_seeds = rng.randint(0, 2**32, size=256)
    rotate90rng = np.random.RandomState(rng_seeds[1])
    t_rotate90 = rotate90()
    rotaterng = np.random.RandomState(rng_seeds[2])
    t_rotate = rotate(torch.pi / 18)
    fliprng_h = np.random.RandomState(rng_seeds[3])
    t_flip_h = flip((2,))
    fliprng_v = np.random.RandomState(rng_seeds[4])
    t_flip_v = flip((1,))
    croprng = np.random.RandomState(rng_seeds[0])
    t_cropper = crop(0.1, 0.1)

    noiserng = np.random.RandomState(rng_seeds[5])
    noisegen = torch.Generator().manual_seed(int(rng_seeds[6]))
    t_gnoise = noise(0, 0.1, gen=noisegen)

    blurrng = np.random.RandomState(rng_seeds[8])

    noiserng_hr = np.random.RandomState(rng_seeds[7])
    noisegen_hr = torch.Generator().manual_seed(int(rng_seeds[8]))
    t_gnoise_hr = noise(0, 0.1, gen=noisegen_hr)


    def _inner(images, low_res, high_res = None,
):
        with torch.no_grad():
            augmented_images_lr = list()
            augmented_images_hr = None if high_res is None else list()
            augmented_images_meta = list()

            for image in images:
                augmented_image_meta = list()

                if rotate90rng.rand() < rotate90_p:
                    image, meta = t_rotate90(image, rotate90rng)
                    augmented_image_meta.append(meta)

                if fliprng_h.rand() < flip_p_h:
                    image, meta = t_flip_h(image, fliprng_h)
                    augmented_image_meta.append(meta)

                if fliprng_v.rand() < flip_p_v:
                    image, meta = t_flip_v(image, fliprng_v)
                    augmented_image_meta.append(meta)

                if rotaterng.rand() < rotate_p:
                    image, meta = t_rotate(image, rotaterng)
                    augmented_image_meta.append(meta)

                if croprng.rand() < crop_p:
                    image, meta = t_cropper(image, croprng)
                    augmented_image_meta.append(meta)

                # interpolate low_res

                image_lr = nn.functional.interpolate(
                    image.unsqueeze(0), size=low_res, mode='bilinear', align_corners=False
                ).squeeze(0)

                # pixel transforms low_res

                if noiserng.rand() < noise_p:
                    image_lr = t_gnoise(image_lr, noiserng)

                if blurrng.rand() < blur_p:
                    image_lr = tvt.gaussian_blur(image_lr, kernel_size=(3, 3), sigma=(0.5, 1.0))

                augmented_images_lr.append(image_lr)


                if high_res is not None:

                    # interpolate high_res

                    image_hr = nn.functional.interpolate(
                        image.unsqueeze(0), size=high_res, mode='bilinear', align_corners=False
                    ).squeeze(0)

                    # pixel transforms low_res

                    if noiserng_hr.rand() < noise_p:
                        image_hr = t_gnoise_hr(image_hr, noiserng)

                    if blurrng.rand() < blur_p:
                        image_hr = tvt.gaussian_blur(image_hr, kernel_size=(3, 3), sigma=(0.5, 1.0))

                    augmented_images_hr.append(image_hr)

                augmented_images_meta.append(augmented_image_meta)

            return augmented_images_lr, augmented_images_hr, augmented_images_meta

    return _inner



def np_rotate90():

    def _inner(image, rng):
        rotations = rng.randint(0, 4)
        rotate_image = np.rot90(image, rotations, axes=(1, 2))
        rotate_image_meta = {"rotate90:", rotations}
        return rotate_image, rotate_image_meta

    return _inner



def np_flip(axis):

    def _inner(image, rng):
        flip_image = np.flip(image, axis)
        flip_image_meta = {"flip": axis}
        return flip_image, flip_image_meta

    return _inner



def np_crop(vertical_param, horizontal_param):

    def _inner(image, rng):
        cols, rows = image.shape[1:]
        top = rng.randint(0, rows * vertical_param)
        bottom = rng.randint(0, rows * vertical_param)
        left = rng.randint(0, cols * horizontal_param)
        right = rng.randint(0, cols * horizontal_param)
        cropped_image = image[..., top:rows - bottom, left:cols - right]
        cropped_image_meta = {"crop:", (top, bottom, left, right)}
        return cropped_image, cropped_image_meta

    return _inner



def augment_light_v0_1_numpy(
        rng,
        rotate90_p = 0.5,
        flip_p_h = 0.5,
        flip_p_v = 0.2,
        rotate_p = 0.5,
        crop_p = 1.0,
):
    t_rotate90 = np_rotate90()
    rotate90rng = np.random.RandomState(rng.randint(0, 2**32))
    t_rotate = rotate(torch.pi / 18)
    rotaterng = np.random.RandomState(rng.randint(0, 2**32))
    t_flip_h = np_flip((2,))
    fliprng_h = np.random.RandomState(rng.randint(0, 2**32))
    t_flip_v = np_flip((1,))
    fliprng_v = np.random.RandomState(rng.randint(0, 2**32))
    t_cropper = crop(0.1, 0.1)
    croprng = np.random.RandomState(rng.randint(0, 2**32))


    def _inner(images):
        augmented_images = list()
        augmented_images_meta = list()

        t0 = time.time()
        for image in images:
            augmented_image_meta = list()

            if rotate90rng.rand() < rotate90_p:
                image, meta = t_rotate90(image, rng)
                augmented_image_meta.append(meta)

            if fliprng_h.rand() < flip_p_h:
                image, meta = t_flip_h(image, rng)
                augmented_image_meta.append(meta)

            if fliprng_v.rand() < flip_p_v:
                image, meta = t_flip_v(image, rng)
                augmented_image_meta.append(meta)

            if rotaterng.rand() < rotate_p:
                image, meta = t_rotate(image, rng)
                augmented_image_meta.append(meta)

            if croprng.rand() < crop_p:
                image, meta = t_cropper(image, rng)
                augmented_image_meta.append(meta)

            augmented_images.append(image)
            augmented_images_meta.append(augmented_image_meta)
        print(f"Preprocessing time: {time.time() - t0}")

        return augmented_images, augmented_images_meta

    return _inner
