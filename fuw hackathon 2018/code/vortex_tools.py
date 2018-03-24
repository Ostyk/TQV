import numpy as np
import scipy.ndimage.measurements as sci_meas
import holoviews as hv
import peakutils as pu
from scipy import ndimage
from scipy.ndimage import filters
import xarray as xr
import param
from holoviews.operation import Operation

import vortex_tools_core as vtc
import fringe_contrast as fc


intensity_dim = hv.Dimension("Intensity", unit="arb.u.")


def fourier_transform(image, return_raw_transform=True, transform_size=vtc.DEF_TRANSFORM_SIZE):
    """
    Calculates the Fourier transform of a holoviews.Raster and returns another holoviews.Raster containing the
    absolute value with logarithmic scaling.

    :param image: numpy array or holoviews.Element2D to be Fourier transformed
    :param return_raw_transform: whether the (complex) raw transform should be returned (default) or only the
    image
    :param transform_size: Size to which the transform is cut. Defaults to 200, cutting can be avoided by
    setting
    transform_size=0.
    :return: (image_of_transform, transform) if return_raw_transform=True is set, otherwise just
    image_of_transform
    (holoviews.Raster)
    """

    if isinstance(image, hv.Element2D):
        data = image.dimension_values(2, flat=False)[::-1]
    else:
        data = image

    transform, transform_abs = vtc.fourier_transform(data, transform_size=transform_size)

    y_size, x_size = transform_abs.shape
    if isinstance(image, hv.Element2D):
        x_coords = image.dimension_values(0)
        y_coords = image.dimension_values(1)
        y_size, x_size = transform_abs.shape
        x_freq = 1 / (x_coords[-1] - x_coords[0])
        y_freq = 1 / (y_coords[-1] - y_coords[0])
        x2 = (np.arange(x_size) - x_size // 2) * x_freq
        y2 = (np.arange(y_size) - y_size // 2) * y_freq
    else:
        x2 = np.arange(x_size)
        y2 = np.arange(y_size)
    dataset = hv.Dataset((x2, y2, transform_abs[::-1]), ["x", "y"], "Fourier transform")

    plot_options = {'Image': dict(logz=True, colorbar=True)}
    style_options = {'Image': dict(cmap='inferno')}
    image = dataset.to(hv.Image)(plot=plot_options, style=style_options)

    if not return_raw_transform:
        return image
    return image, transform


def inv_fourier_transform(image, return_raw_transform=False, offset=(0, 0)):
    """
    Calculates the inverse Fourier transform of a Numpy array and returns a holoviews.HoloMap with the kdim
    containing
    "Absolute value" and "Phase" of the transform.

    :param image: numpy array or holoviews.Element2D containing the data to be transformed
    :param return_raw_transform: whether the raw transform should be returned along with the plots (default is
    False)
    :return: A HoloMap containing "Absolute value" and "Phase" of the
    """

    if isinstance(image, hv.Element2D):
        data = image.dimension_values(2, flat=False)[::-1]
    else:
        data = image

    transform, transform_abs, transform_phase = vtc.inv_fourier_transform(data)

    y_size, x_size = transform_abs.shape
    if isinstance(image, hv.Element2D):
        x_coords = image.dimension_values(0)
        y_coords = image.dimension_values(1)
        x_freq = 1 / (x_coords[-1] - x_coords[0])
        y_freq = 1 / (y_coords[-1] - y_coords[0])
        x2 = (np.arange(x_size) - x_size // 2) * x_freq
        y2 = (np.arange(y_size) - y_size // 2) * y_freq
    else:
        x2 = np.arange(x_size)
        y2 = np.arange(y_size)

    x2 += offset[0]
    y2 += offset[1]

    plot_opts = {'Image': dict(colorbar=True)}
    hmap = hv.HoloMap(kdims=["Dimension"])
    dataset_abs = hv.Dataset((x2, y2, transform_abs[::-1]), ["x", "y"], vdims=[intensity_dim],
                             label="Inverse Fourier transform")
    hmap["Absolute value"] = dataset_abs.to(hv.Image)(style=dict(Image=dict(cmap="gray")), plot=plot_opts)
    #hmap["Absolute value"] = hv.Raster(transform_abs, label="Inverse Fourier transform",
    #                                   vdims=[intensity_dim])(
    #    style={'Raster': dict(cmap='gray')}, plot={'Raster': dict(colorbar=True)})
    dataset_phase = hv.Dataset((x2, y2, transform_phase[::-1]), ["x", "y"], vdims=[intensity_dim],
                               label="Inverse Fourier transform")
    hmap["Phase"] = dataset_phase.to(hv.Image)(style=dict(Image=dict(cmap="viridis")), plot=plot_opts)
    #hmap["Phase"] = hv.Raster(transform_phase, label="Inverse Fourier transform",
    #                          vdims=[hv.Dimension("Phase", unit="$\pi$")])(
    #    style={'Raster': dict(cmap='viridis')}, plot={'Raster': dict(colorbar=True)})
    if not return_raw_transform:
        return hmap
    return hmap, transform


def convert_blob_coordinates(blob, image):
    x , y = image.matrix2sheet(blob[0], blob[1])
    dia, _ = image.matrix2sheet(blob[0], blob[1] - blob[2])
    dia -= x
    dia *= 2

    return (x, y, dia)


def extract_phase(interferogram, min_sigma=vtc.DEF_MIN_SIGMA, max_sigma=vtc.DEF_MAX_SIGMA,
                  overlap=vtc.DEF_OVERLAP, threshold=vtc.DEF_THRESHOLD, method=vtc.DEF_METHOD,
                  transform_size=vtc.DEF_TRANSFORM_SIZE, auto=True, silent=False,
                  number=vtc.DEF_NUMBER_BLOBS):
    """
    Extracts the phase of an interferogram by calculating the Fourier transform, shifting the outer blob to
    the center
    of the frequency space and performing an inverse Fourier transform. The blob is found automatically.
    Returns a
    report showing the Fourier transform with the found blobs, as well as a HoloMap containing the inverse
    transform.

    :param interferogram: holoviews.Raster containing an interferogram
    :param min_sigma: For find_blobs
    :param max_sigma: For find_blobs
    :param overlap: For find_blobs
    :param threshold: For find_blob
    :param method: Peak finding method: 'log' for Laplacian of Gaussians, 'dog' for Difference of Gaussians
    (Default: 'log')
    :param transform_size: The size in pixels to which the Fourier transform should be clipped
    :return: (report, back_transform) holoviews.Overlay, holoviews.HoloMap
    """

    if isinstance(interferogram, hv.HoloMap):
        reports = interferogram.clone(shared_data=False)
        inv_transforms = interferogram.clone(shared_data=False)
        phases = interferogram.clone(shared_data=False)
        raw_inv_transforms = interferogram.clone(shared_data=False)
        for key, element in interferogram.items():
            print("Extracting {}".format(key))
            res = extract_phase(element, min_sigma, max_sigma, overlap, threshold,
                                                 method, transform_size, auto, True, number)
            reports[key] = res["report"]
            if res["inv_transform"] == hv.Empty:
                inv_transforms[key] = hv.RGB(data=np.zeros((200, 200)))
                phases[key] = hv.Image(data=np.zeros((200, 200)))
                raw_inv_transforms[key] = hv.Image(data=np.zeros((200, 200)))
            else:
                x_offset = np.average(element.range(0))
                y_offset = np.average(element.range(1))
                abs_map = res["inv_transform"]["Absolute value"]
                phase_map = res["inv_transform"]["Phase"]
                inv_transforms[key] = plot_rgb_phases(abs_map, phase_map, offset=(x_offset, y_offset))
                raw_inv_transforms[key] = abs_map
                phases[key] = phase_map

        return dict(report=reports, inv_transform=inv_transforms, phase=phases,
                    raw_inv_transform=raw_inv_transforms)


    transform_plot, transform = fourier_transform(interferogram, transform_size=transform_size)

    # find and plot blobs
    if auto:
        blobs = vtc.find_number_blobs(transform, number=number, min_sigma=min_sigma, max_sigma=max_sigma,
                                      overlap=overlap,
                                      threshold=threshold, method=method)
    else:
        blobs = vtc.find_blobs(transform, max_sigma=max_sigma, min_sigma=min_sigma,
                               overlap=overlap, threshold=threshold, method=method)
    transform_overlay = transform_plot
    for blob in blobs:
        #transform_overlay *= hv.Ellipse(blob[1], blob[0], blob[2] * 2)(
        transform_overlay *= hv.Ellipse(*convert_blob_coordinates(blob, transform_plot))(
            style={'Ellipse': dict(color='blue')})
    transform_overlay = transform_overlay.relabel("Blob detection")

    if not blobs.shape[0] == number:
        if not silent:
            print("Error: found {} blobs instead of {}!".format(blobs.shape[0], number))
        if not isinstance(transform_overlay, hv.Overlay):
            transform_overlay = hv.Overlay(items=[transform_overlay])
        return dict(report=transform_overlay, inv_transform=hv.Empty, blobs=tuple())

    main_blob = vtc.pick_blob(blobs)
    if not silent:
        print("Detected blob position:", main_blob)

    # extend the main blob for greatest possible resolution:
    center = (transform_size / 2) - 1
    main_blob[2] = np.sqrt((main_blob[0] - center)**2 + (main_blob[1] - center)**2) / 2

    #transform_overlay *= hv.Ellipse(main_blob[1], main_blob[0], main_blob[2] * 2)(
    transform_overlay *= hv.Ellipse(*convert_blob_coordinates(main_blob, transform_plot))(
        style={'Ellipse': dict(color='green')})
    transform_overlay = transform_overlay.relabel("Blob detection")

    report = transform_overlay

    shifted_transform = vtc.mask_and_shift(transform, main_blob[0], main_blob[1], main_blob[2])
    shifted_transform_abs = np.abs(shifted_transform) + np.median(np.abs(transform))
    # shifted_transform_plot = hv.Image(shifted_transform_abs,
    #                                    label="Masked and shifted transform")(
    #     plot={'Image': dict(logz=True, colorbar=True)},
    #     style={'Image': dict(cmap='inferno')})
    # report += shifted_transform_plot

    # calculate inverse fourier transform and make sure coordinates match with original image
    x_offset = np.average(interferogram.range(0))
    y_offset = np.average(interferogram.range(1))
    inv_transform = inv_fourier_transform(transform_plot.clone(data=shifted_transform),
                                          offset=(x_offset, y_offset))

    if not silent:
        print("final transform center of mass (log):", sci_meas.center_of_mass(np.log(shifted_transform_abs)))
        print("final transform center of mass (lin):", sci_meas.center_of_mass(shifted_transform_abs))

    return dict(report=report, inv_transform=inv_transform, blobs=blobs)


def extract_phase_from_coordinates(interferogram, blob, transform_size=vtc.DEF_TRANSFORM_SIZE):

    transform_plot, transform = fourier_transform(interferogram, transform_size=transform_size)

    transform_overlay = transform_plot
    transform_overlay = transform_overlay.relabel("Blob detection")
    transform_overlay *= hv.Ellipse(blob[1], blob[0], blob[2] * 2)(
        style={'Ellipse': dict(color='green')})
    transform_overlay = transform_overlay.relabel("Blob detection")

    report = hv.Layout(transform_overlay)

    shifted_transform = vtc.mask_and_shift(transform, blob[0], blob[1], blob[2])
    back_transform = inv_fourier_transform(shifted_transform)

    return report, back_transform


def plot_rgb_phases(abs_map, phase_map , offset=(0, 0)):
    """
    Plots a visualization of an inverse Fourier transform, where the absolute value is plotted as brightness
    and the phase is plotted as color.

    :param transform_map: A HoloMap as provided by inv_fourier_transform
    :return: holoviews.RGB, representing the complex plane
    """
    absolute = abs_map.dimension_values(2, flat=False)[::-1]
    phase = phase_map.dimension_values(2, flat=False)[::-1]
    red = 0.5 * (np.sin(phase * np.pi) + 1) * absolute / absolute.max()
    green = 0.5 * (np.sin(phase * np.pi + 2 / 3 * np.pi) + 1) * absolute / absolute.max()
    blue = 0.5 * (np.sin(phase * np.pi + 4 / 3 * np.pi) + 1) * absolute / absolute.max()
    x_min, x_max = abs_map.range(0)
    y_min, y_max = abs_map.range(1)

    return hv.RGB(np.dstack([red, green, blue]),
                  label="Absolute value (brightness) and phase (color)",
                  #bounds=(0, absolute.shape[0], absolute.shape[1], 0))
                  bounds=(x_min, y_min, x_max, y_max))


def plot_overlay_phases(abs_map, phase_map):
    """
    Plots a visialization of an inverse Fourier transform, where a semi-transparent color map is overlaid
    onto a grayscale image of the absolute value.

    :param transform_map: A HoloMap as provided by inv_fourier_transform
    :return: holoviews.Overlay
    """
    absolute = abs_map.relabel("Absolute value")
    phase = phase_map.relabel("Phase")(style={'Image': dict(alpha=0.3), 'Image': dict(alpha=0.3)})
    return (absolute * phase).relabel("Absolute value (brightness) and phase (color)")


def phase_plot(abs_map, phase_map):
    """
    Returns a layout of the combination of plot_overlay_phases and plot_rgb_phases

    :param transform_map: A HoloMap as provided by inv_fourier_transform
    :return: holoviews.Layout
    """
    return plot_overlay_phases(abs_map, phase_map) + plot_rgb_phases(abs_map, phase_map) + phase_map.relabel("Phase")


def complex_plot(array):
    """
    Plots the absolute value and phase of a 2d complex array

    :param array: numpy array
    :return: Layout of holoviews.Raster elements
    """
    plot = {'Raster': dict(colorbar=True)}
    absolute = hv.Raster(np.abs(array), label="Absolute value")(style={'Raster': dict(cmap='plasma')}, plot=plot)
    phase = hv.Raster(np.angle(array)/np.pi, label="Phase", vdims=[hv.Dimension("Phase", unit="($\pi$)")])(
        style={'Raster': dict(cmap='viridis')}, plot=plot)

    return absolute + phase


def phase_shift(obj, value):  # provide Raster and phase value in units of pi
    """
    Shifts the phase of a holoviews.Raster by a given value, assuming that both are in the units of pi.

    :param raster: holoviews.Raster
    :param value: Shift value in pi (0 will not change anything)
    :return: a new, shifted Raster
    """
    if isinstance(obj, hv.HoloMap):
        res_map = obj.clone(shared_data=False)
        for key, image in obj.items():
            res_map[key] = phase_shift(image, value)
        return res_map
    else:
        data = obj.dimension_values(2, flat=False)[::-1]
        data = (data + 1 + value) % 2 - 1
        return obj.clone(data=data)


def phase_shift_map(raster, steps=8):
    """
    Returns a holoviews.HoloMap with different phase shifted versions of a holoviews.Raster
    :param raster: The raster to be shifted
    :param steps: How many different phase offsets should be shown (default: 5)
    :return: A HoloMap of Rasters
    """
    shifted_rasters = {phase: phase_shift(raster, phase) for phase in
                       np.linspace(-1, 1, int(steps), endpoint=False)}
    return hv.HoloMap(shifted_rasters, kdims=["Phase shift"])


def fringe_plot(section, thresh, min_dist=4):
    intensity = section.dimension_values('Intensity')
    peakind = pu.indexes(intensity, thres=thresh, min_dist=min_dist)
    peak_num = len(peakind)
    peaks = hv.Scatter((section.dimension_values('x')[peakind],
                        section.dimension_values('Intensity')[peakind]),
                    label="{} peaks".format(peak_num))

    return section.opts({'Curve':{'style':dict(marker='x')}}) * \
        peaks.opts({'Scatter':{'style':dict(color='red')}}), peakind


def fft_blur(image, sigma=vtc.DEF_BLUR):
    data = vtc.fft_blur(image.data, sigma)
    return image.clone(data=data)


def homogenize(image, sigma=vtc.DEF_SIGMA, blur=vtc.DEF_BLUR_HOMO):
    if isinstance(image, hv.HoloMap):
        hmap = image.clone(shared_data=False)
        for key, im in image.items():
            im = homogenize(im, blur=blur)
            hmap[key] = im
        return hmap
    else:
        data = vtc.homogenize(image.data, sigma, blur)
        image = image.clone(data=data)
        image = image.relabel(image.label + " (homogenized)")
        return image


class homogenize_op(Operation):

    sigma = param.Number(default=vtc.DEF_SIGMA)

    blur = param.Number(default=vtc.DEF_BLUR_HOMO)

    label = param.String(default=" (homogenized)")

    def _process(self, element, key=None):
        data = element.dimension_values(2, flat=False)[::-1]
        element = element.clone(data=vtc.homogenize(data, self.p.sigma, self.p.blur))
        element = element.relabel(element.label + self.p.label)
        return element


class apply_window(Operation):

    window = param.String(default=vtc.DEF_WINDOW)

    args = param.List(default=[])

    label = param.String(default=" (windowed)")

    def _process(self, element, key=None):
        data = element.dimension_values(2, flat=False)[::-1]
        data = vtc.apply_window(data, self.p.window, *(self.p.args))
        element = element.clone(data=data).relabel(element.label + self.p.label)
        return element


def phase_shift_from_pixel(obj, pixel=None, area=None):
    # specify pixel as (x, y)
    # specify area as (center_x, center,_y, size)

    if isinstance(obj, hv.HoloMap):
        res_map = obj.clone(shared_data=False)
        for key, image in obj.items():
            res_map[key] = phase_shift_from_pixel(image, pixel, area)
        return res_map
    else:
        if pixel is None and area is None:
            xsize, ysize = obj.dimension_values(2, flat=False).shape
            pixel = (int(xsize/2), int(ysize/2))
        if pixel is not None:
            phase = obj[obj.matrix2sheet(*pixel)]
        else:
            x, y, size = area
            halfsize = int(size / 2)
            left = x - halfsize
            right = x + halfsize
            top = y + halfsize
            bottom = y - halfsize
            phase = np.sum(obj[left:right, bottom:top].data)
        return phase_shift(obj, -phase).relabel(obj.label + " (shifted)")


def find_peaks(data, threshold=None, neighborhood_size=5):

    data = data.data.copy()

    if threshold is None:
        threshold = data.max() / 3

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

    return x, y


def find_vortices(phase, inv_transform, threshold=None, neighborhood_size=5,
                  mask_threshold=0.05):

    # prepare the mask
    transf = np.sum(inv_transform.data, axis=2)
    transf /= transf.max()
    mask = np.zeros_like(transf)
    mask[transf > mask_threshold] = 1
    mask = ndimage.binary_dilation(mask)

    # calculate the laplacian and apply the mask
    phase_cos = np.cos(np.pi*phase.data)
    laplace = np.abs(ndimage.laplace(phase_cos))
    laplace[mask == 0] = 0

    return find_peaks(hv.Raster(laplace), threshold, neighborhood_size)


def plot_vortices(phase, peaks):
    if peaks is None:
        return phase
    style_options = dict(Scatter=dict(style=dict(color='none', edgecolor='red', s=100)))
    scatter = hv.Scatter(peaks).opts(style_options)
    return (phase * scatter).relabel(phase.label + " ({} vortices)".format(len(peaks[0])))


class fringe_contrast(Operation):

    label = param.String(default=" (fringe contrast)")

    maxval = param.Number(default=fc.DEF_MAXVAL)

    tilesize = param.Number(default=fc.DEF_TILESIZE)

    def _process(self, element, key=None):
        data = element.dimension_values(2, flat=False)[::-1]
        contrasts = vtc.fringe_contrast_map(data, self.p.tilesize, self.p.maxval)

        x_axis = element.dimension_values(0, expanded=False)
        x_axis = np.concatenate((np.zeros(self.p.tilesize//2), x_axis))
        x_axis = x_axis[::self.p.tilesize][1:contrasts.shape[1]+1]
        y_axis = element.dimension_values(1, expanded=False)[::-1]
        y_axis = np.concatenate((np.zeros(self.p.tilesize//2), y_axis))
        y_axis = y_axis[::self.p.tilesize][1:contrasts.shape[0]+1]

        return hv.Dataset((x_axis, y_axis, contrasts), ['x', 'y'], "Contrast").to(hv.Image)


class highpass(Operation):

    label = param.String(default=" (high-pass)")

    sigma = param.Number(default=5)

    def _process(self, element, key=None):
        data = element.dimension_values(2, flat=False)[::-1]
        lowpass = ndimage.gaussian_filter(data, self.p.sigma)
        highpass = data - lowpass
        highpass -= highpass.min()
        element = element.clone(data=highpass)
        element = element.relabel(element.label + self.p.label)
        return element
