from scipy import ndimage
from scipy.ndimage import filters

def find_peaks(image, threshold=None, neighborhood_size=5): 
    
    data = image.dimension_values(2, flat=False)[::-1]
    
    if threshold is None:
        threshold = data.max() / 2
    
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    
    res = [image.matrix2sheet(*xy) for xy in zip(y, x)]
    return res


def find_vortices(phase, inv_transform, threshold=None, neighborhood_size=5,
                  mask_threshold=0.1, dilation=3, return_mask=False):
    
    # prepare the mask
    mask_data = np.sum(np.dstack([inv_transform.dimension_values(i, flat=False)[::-1] for i in [2, 3, 4]]), axis=2)
    transf = mask_data / mask_data.max()
    mask = np.zeros_like(transf)
    mask[transf > mask_threshold] = 1
    mask = ndimage.binary_dilation(mask, iterations=dilation)
    mask = ndimage.binary_fill_holes(mask)
    
    
    # calculate the laplacian and apply the mask
    phase_data = phase.dimension_values(2, flat=False)[::-1]
    phase_cos = np.cos(np.pi*phase_data)
    laplace = np.abs(ndimage.laplace(phase_cos))
    laplace[mask == 0] = 0
    
    if return_mask:
        return find_peaks(phase.clone(data=laplace), threshold, neighborhood_size), phase.clone(data=mask)
    else:
        return find_peaks(phase.clone(data=laplace), threshold, neighborhood_size)


def plot_vortices(phase, peaks):
    if peaks is None:
        return phase
    style_options = dict(Scatter=dict(style=dict(color='none', edgecolor='red', s=100, alpha=0.5)))
    scatter = hv.Scatter(peaks).opts(style_options)
    overlay = (phase * scatter).relabel(phase.label + " ({} vortices)".format(len(peaks)))
    overlay = overlay.opts(dict(Overlay=dict(plot=dict(aspect="equal"))))
    return overlay