import numpy as np
from scipy import ndimage


"""
Python implementation of Tom Mertens's pyramid blending MATLAB code.

Copyright (c) 2015, Tom Mertens
All rights reserved.

Rewritten in 2022, Thomas Eboli, Centre Borelli, ENS Paris-Saclay

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


def multiscale_blending(seq, W, n_scales):
    if n_scales == 0:
        auto_ref = True
    else:
        auto_ref = False
    if n_scales == -1:
        auto_min = True
    else:
        auto_min = False
    if n_scales == -2:
        auto_max = True
    else:
        auto_max = False

    n, h, w = seq.shape

    # automatic setting of the number of scales
    if auto_ref or auto_min or auto_max:
        n_scale_ref = int(np.floor(np.log(min(h, w)) / np.log(2)))

        n_sacles = 1
        hp = h
        wp = w

        while (auto_ref and (n_scales < n_scale_ref)) or (auto_min and (hp > 1 and wp > 1)) or (auto_max and (hp > 1 or wp > 1)):
            n_scales = n_scales + 1
            hp = int(np.ceil(hp / 2))
            wp = int(np.ceil(wp / 2))

    # allocate memory for pyr
    pyr = []
    hp = h
    wp = w
    for scale in range(n_scales):
        pyr.append(np.zeros((hp, wp), dtype=seq.dtype))
        hp = int(np.ceil(hp / 2))
        wp = int(np.ceil(wp / 2))

    # multiresolution blending
    filter1d = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=seq.dtype)
    for i in range(n):
        # construct the pyramids for each image
        pyrW = gaussian_pyramid(W[i], filter1d, n_scales)  # tuple of (hp,wp) arrays
        pyrI = laplacian_pyramid(seq[i], filter1d, n_scales)

        # blend
        for scale in range(n_scales):
            pyr[scale] = pyr[scale] + pyrW[scale] * pyrI[scale]

    # reconstruct
    out = pyr[-1]
    for l in range(n_scales-2, -1, -1):
        odd = 2 * np.array(out.shape[-2:]) - np.array(pyr[l].shape[-2:])
        out = pyr[l] + upsample(out, odd, filter1d)

    return out


def upsample(I, odd, filter1d):
    # increase resolution
    I = np.pad(I, [(1, 1), (1, 1)], mode='edge')
    r = 2 * I.shape[-2]
    c = 2 * I.shape[-1]
    R = np.zeros((r, c), dtype=I.dtype)
    R[0:r:2, 0:c:2] = 4 * I

    # interpolate, convolve with separable filter
    R = ndimage.convolve1d(R, filter1d, axis=-1)  # horizontal pass
    R = ndimage.convolve1d(R, filter1d, axis=-2)  # vertical pass

    # remove the border
    R = R[2:r-2-odd[0], 2:c-2-odd[1]]

    return R


def downsample(I, filter1d):
	# low pass, convolve with separable filter
	R = ndimage.convolve1d(I, filter1d, axis=-1, mode='reflect')  # horizontal pass
	R = ndimage.convolve1d(R, filter1d, axis=-2, mode='reflect')  # vertical pass

	# decimate
	R = R[0::2, 0::2]

	return R


def gaussian_pyramid(I, filter1d, n_level=None):
	r = I.shape[0]
	c = I.shape[1]

	if n_level is None:
		# compute the highest level possible in the pyramid
		n_level = int(np.floor(np.log(min(r, c)) / np.log(2)))

	# start by copying the image to the finest level
	pyr = [I]

	# recursively downsample the image
	for l in range(1, n_level):
		I = downsample(I, filter1d)
		pyr.append(I)

	return tuple(pyr)


def laplacian_pyramid(I, filter1d, n_level=None):
	r = I.shape[0]
	c = I.shape[1]

	if n_level is None:
		# compute the highest level possible in the pyramid
		n_level = int(np.floor(np.log(min(r, c)) / np.log(2)))

	# start by copying the image to the finest level
	pyr = []
	J = I 
	for l in range(n_level - 1):
		# apply low pass filter, and downsample
		I = downsample(J, filter1d)
		odd = 2 * np.array(I.shape[-2:]) - np.array(J.shape[-2:])  # for each dimensaion, check
		# if the upsampled verseion has to be odd in each level, store the difference between
		# timage and upsampled low pass version
		pyr.append(J - upsample(I, odd, filter1d))
		J = I  # continue with the low pass image

	pyr.append(J)  # the coarsest level contains the residual low pass image

	return tuple(pyr)
