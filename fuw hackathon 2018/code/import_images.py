import PIL.Image
import holoviews as hv
import numpy as np
import re
import io
import gzip
import os
import time
from holoviews.plotting.mpl import MPLRenderer
import imageio

power_dim = hv.Dimension("Power", unit="µW")
intensity_dim = hv.Dimension("Intensity", unit="arb.u.")


import os
import xarray as xr

def import_image(filename, px_per_unit=None, unit=""):

    image_array = np.rot90(np.array(Image.open(filename).convert('L'), dtype='float'), 3)
    
    lx, ly = image_array.shape
    if px_per_unit is not None:
        x_span = lx / px_per_unit / 2
        y_span = ly / px_per_unit / 2
    else:
        x_span = lx / 2
        y_span = ly / 2
        unit = "px"
    x_coords = np.linspace(-x_span , x_span, lx)
    y_coords = np.linspace(-y_span , y_span, ly)
    
    data_array = xr.DataArray(image_array, coords=[("x", x_coords), ("y", y_coords)])
    data_array.name = "intensity"
    data_array.attrs['units'] = 'arb. u.'
    
    return data_array


def import_image_number(folder, number, px_per_unit=None, unit=""):
    pattern = re.compile(r"^.*?(\d+)(?:.\w+)?$")
    
    for filename in [f for f in os.listdir(folder) if is_image(f)]:
        match = pattern.match(filename)
        if match is not None:
            file_number = int(match.group(1))
            if number == file_number:
                break
    else:
        raise ValueError("File number {} not found!".format(number))
            
    return import_image(os.path.join(folder, filename), px_per_unit=px_per_unit, unit=unit)


def is_image(filename):
    extensions = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    for extension in extensions:
        if filename.endswith(extension):
            return True
    return False


def import_image_folder(folder, px_per_unit=None, unit="", label="Interferogram"):
    pattern = re.compile(r"^.*?(\d+)(?:.\w+)?$")

    res_map = hv.HoloMap(kdims=["Number"])

    for filename in [f for f in os.listdir(folder) if is_image(f)]:
        match = pattern.match(filename)
        if match is not None:
            number = int(match.group(1))
            res_map[number] = import_image(os.path.join(folder, filename), px_per_unit, unit)
    
    return res_map




def save_hmap_video(hmap, filename, fps=5):
    start_time = time.time()
    renderer = MPLRenderer.instance(holomap="mp4", fps=fps)
    with io.BufferedWriter(open(filename, "wb")) as file:
        file.write(renderer(hmap)[0])
    elapsed_time = time.time() - start_time
    print("Saved {} ({:.2f} s)".format(filename, elapsed_time))


def resize_frame(arr, size):
    frame_size = arr.shape[0:2]
    if frame_size[0] > size[0] or frame_size[1] > size[1]:
        arr = arr[0:size[0], 0:size[1]]
    elif frame_size[0] < size[0] or frame_size[1] < size[1]:
        temp = np.ones(size[0], size[1], 4)
        temp[0:frame_size[0], 0:frame_size[1]] = arr
        arr = temp
    return arr


def make_video_from_hmap(hmap, name, fps=2):
    renderer = hv.renderer('matplotlib')
    writer = imageio.get_writer(name, fps=fps)
    size = None
    
    if isinstance(hmap, hv.Layout) or isinstance(hmap, hv.NdLayout):
        keys = hmap[hmap.keys()[0]].keys()
        for key in keys:
            print("writing key {}".format(key))
            dim = hmap.dimensions()[1]
            label = dim.name + ": " + dim.pprint_value(key)
            if dim.unit != None:
                label += " " + dim.unit
            item = hmap[:, key]
            item = item.relabel(label)
            hv.renderer('matplotlib').save(item, '_temp', fmt='png')
            img = PIL.Image.open("_temp.png")
            arr = np.array(img)
            if size == None:
                size = arr.shape[0:2]
            else:
                arr = resize_frame(arr, size)
            writer.append_data(arr)

    else:
        for key, item in hmap.items():
            print("writing key {}".format(key))
            label = ""
            if not isinstance(key, tuple):
                key = tuple([key])
            for value, dim in zip(key, hmap.dimensions()):
                dim_label = dim.name + ": " + dim.pprint_value(value)
                if dim.unit != None:
                    dim_label += " " + dim.unit
                if label == "":
                    label = dim_label
                else:
                    label += ", " + dim_label
            item = item.relabel(label)
            hv.renderer('matplotlib').save(item, '_temp', fmt='png')
            img = PIL.Image.open("_temp.png")
            arr = np.array(img)
            if size == None:
                size = arr.shape[0:2]
            else:
                arr = resize_frame(arr, size)
            writer.append_data(arr)
    writer.close()
    os.remove("_temp.png")


def save_pickle(obj, filename):
    start_time = time.time()
    with io.BufferedWriter(gzip.open(filename, "wb", compresslevel=5)) as file:
        hv.Store.dump(obj, file)
    elapsed_time = time.time() - start_time
    print("Saved {} ({:.2f} s)".format(filename, elapsed_time))


def load_pickle(filename):
    start_time = time.time()
    with io.BufferedReader(gzip.open(filename, "rb")) as file:
        obj = hv.Store.load(file)
    elapsed_time = time.time() - start_time
    print("Loaded {} ({:.2f} s)".format(filename, elapsed_time))
    return obj
