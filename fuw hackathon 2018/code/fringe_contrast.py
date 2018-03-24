import numpy as np


DEF_MAXVAL = 255
DEF_TILESIZE = 10


def idib(image, maxv=DEF_MAXVAL):
    mu = int(np.average(image))
    
    hist = np.bincount(image.flatten(), minlength=maxv+1)
    
    hist_d = hist[:mu+1]
    weight_d = np.arange(mu+1)
    i_d = np.sum(weight_d*hist_d) / np.sum(hist_d)
    
    hist_b = hist[mu:]
    weight_b = np.arange(mu, maxv+1)
    i_b = np.sum(weight_b*hist_b) / np.sum(hist_b)
    
    return i_d, i_b


def c1_value(image, maxv=DEF_MAXVAL):
    i_d, i_b = idib(image, maxv=maxv+1)
    return (i_b - i_d) / (i_b + i_d)


def iod(image):
    return np.sum(image)


def iodm(image, maxv=DEF_MAXVAL):
    return (maxv - 1) / 2 * np.product(image.shape)


def c2_value(image, maxv=DEF_MAXVAL):
    iod_ = iod(image)
    iodm_ = iodm(image, maxv=maxv)
    
    if iod_ <= iodm_:
        return iod_ / iodm_
    else:
        return 2 - iod_ / iodm_


def c_value(image, maxv=DEF_MAXVAL):
    return c1_value(image, maxv=maxv) * c2_value(image, maxv=maxv)


def blockshaped(arr, tilesize):
    h, w = arr.shape
    nrows = h // tilesize
    ncols = w // tilesize
    arr = arr[: nrows * tilesize, : ncols * tilesize]
    h, w = arr.shape
    return (arr.reshape(h//tilesize, tilesize, -1, tilesize)
               .swapaxes(1,2)
           )


def fringe_contrast_map(image, tilesize=DEF_TILESIZE, maxval=DEF_MAXVAL):
    tiles = blockshaped(image, tilesize)
    contrasts = []
    contrasts = np.zeros(tiles.shape[0:2])
    for x in range(tiles.shape[1]):
        for y in range(tiles.shape[0]):
            contrasts[y, x] = c_value(np.int64(tiles[y, x]))
    return contrasts
