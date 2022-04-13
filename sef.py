import numpy as np
import cv2
from skimage import color

import pyramid_fusion


"""
Python implementation of Charles Hessel's simulated exposure fusion MATLAB code.

Rewritten in 2022, Thomas Eboli, Centre Borelli, ENS Paris-Saclay

"""



def simulated_exposure_fusion(u, M=None, n_scales=0, alpha=4, beta=0.5, lambd=0.125, S_black=1, S_white=99):
    """Single-image tone-mapping with the Exposure Fusion algorithm, based on
    Charles Hessel, IPOL' 19: Simulated Exposure Fusion
    """
    assert(u.dtype == np.float32)

    ## Find the optimal number of images
    if u.ndim == 3:
        z = color.rgb2hsv(u)
        l = z[..., 2]
    else:
        l = u

    c = u / (l[..., None] + 1e-8)

    cval = np.median(u)

    ## Find the right
    if M is not None and M > 0:
        Mp = M - 1
        Ns = int(np.floor(Mp * cval))
        N = Mp - Ns
        Nx = max(N, Ns)
    else:
        Mp = 1
        Ns = int(np.floor(Mp * cval))
        N = Mp - Ns
        Nx = max(N, Ns)
        tmax1 =  (1     + (Ns+1) * (beta-1) / Mp) / (alpha**(1/Nx))
        tmin1s = (-beta + (Ns-1) * (beta-1) / Mp) / (alpha**(1/Nx)) + 1
        tmax0 =  1      + Ns * (beta-1) / Mp
        tmin0 =  1-beta + Ns * (beta-1) / Mp
        while tmax1 < tmin0 or tmax0 < tmin1s:
            Mp = Mp + 1
            Ns = int(np.floor(Mp * cval))
            N = Mp - Ns
            Nx = max(N, Ns)
            tmax1 =  (1     + (Ns+1) * (beta-1) / Mp) / (alpha**(1/Nx))
            tmin1s = (-beta + (Ns-1) * (beta-1) / Mp) / (alpha**(1/Nx)) + 1
            tmax0 =  1      + Ns * (beta-1) / Mp
            tmin0 =  1-beta + Ns * (beta-1) / Mp
            if Mp > 49:
                print('M is limited to 50')
                break
    print('M = %d, with N = %d and Ns = %d.' % (Mp + 1, N, Ns))

    ## Remapping functions
    f = lambda t, k: alpha**(k/Nx) * t
    fs = lambda t, k: alpha**(-k/Nx) * (t-1) + 1

    r = lambda k: (1-beta/2) - (k+Ns)*(1-beta)/Mp
    a = beta / 2 + lambd
    b = beta / 2 - lambd
    g = lambda t, k: (np.abs(t - r(k)) <= beta / 2) * t + \
                     (np.abs(t - r(k)) > beta / 2) * \
                     (np.sign(t - r(k)) * (a - lambd**2 / \
                     (np.abs(t - r(k)) - b + (np.abs(t - r(k)) == b)))  + r(k))

    h  = lambda t, k: g(f(t,  k), k)
    hs = lambda t, k: g(fs(t, k), k)

    dg = lambda t, k: (np.abs(t - r(k)) <= beta / 2) * 1 + \
                      (np.abs(t - r(k)) > beta / 2) * \
                      (lambd**2 / (np.abs(t - r(k)) - b + (np.abs(t - r(k)) == b))**2)


    dh = lambda t, k: alpha**(k/Nx) * dg(f(t, k), k).astype(np.float32)
    dhs = lambda t, k: alpha**(-k/Nx) * dg(fs(t, k), k).astype(np.float32)

    ## Create the sequence of images
    seq = []
    wc = []
    for k in range(-Ns, N+1, 1):
        if k < 0:
            seq.append(hs(l, k))
            wc.append(dhs(l, k))
        else:
            seq.append(h(l, k))
            wc.append(dh(l, k))

    seq = np.stack(seq, axis=0)
    wc = np.stack(wc, axis=0)

    clipsup = seq > 1
    clipinf = seq < 0
    seq[clipsup] = 1
    seq[clipinf] = 0
    wc[clipsup] = 0
    wc[clipinf] = 0

    ## Well-exposedness weights
    we = np.zeros_like(seq)
    for n in range(seq.shape[0]):
        we[n] = np.exp(-0.5 * (seq[n] - 0.5)**2 / 0.2**2)

    ## Final normalized weights
    w = wc * we + 1e-8
    w /= np.sum(w, axis=0, keepdims=True)

    ## Multi-scale blending
    lp = pyramid_fusion.multiscale_blending(seq, w, n_scales)

    # Normalize result
    v = lp[..., None] * c
    v_norm = robust_normalization(v, S_black=S_black, S_white=S_white)

    return v_norm


def rho(image, beta, k, N_star, N, lambd=0.125):
    rho = 1 - beta / 2 - (k + N_star) * (1 - beta) / (N + N_star)
    image_clip = image
    mask = np.abs(image - rho) > beta / 2
    a = beta / 2 + lambd
    b = beta / 2 - lambd
    image_clip[mask] = np.sign(image[mask] - rho) * (a - lambd**2 / (np.abs(image[mask] - b))) + rho

    return image_clip


def robust_normalization(image, S_white=99, S_black=1):
    N = image.shape[0] * image.shape[1]
    im_max = np.max(image, axis=-1)
    im_min = np.min(image, axis=-1)
    im_smax = np.sort(im_max.flatten())
    im_smin = np.sort(im_min.flatten())
    val_max = im_smax.flatten()[int(np.ceil(1 - S_white/100) * N - 1)]
    val_min = im_smin.flatten()[int(np.floor(S_black / 100 * N))]
    image_norm = np.clip((image - val_min) / (val_max - val_min), 0.0, 1.0)

    return image_norm
