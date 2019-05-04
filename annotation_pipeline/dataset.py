import numpy as np
import io


def parse_txt(key, data_stream, func):
    rows = []
    buffer = io.StringIO(data_stream.read().decode('utf-8'))
    while True:
        line = buffer.readline()
        if not line:
            break
        rows.append(func(line))
    return rows


def parse_spectrum_line(s):
    ind_s, mzs_s, int_s = s.split('|')
    return (int(ind_s),
            np.fromstring(mzs_s, sep=' ').astype('float32'),
            np.fromstring(int_s, sep=' '))


def parse_spectrum_coord(s):
    sp_i, x, y = map(int, s.split(','))
    return sp_i, x, y


def real_pixel_indices(spectra_coords):
    coord_pairs = [r[1:] for r in spectra_coords]

    min_x, min_y = np.amin(np.asarray(coord_pairs), axis=0)
    max_x, max_y = np.amax(np.asarray(coord_pairs), axis=0)

    _coord = np.array(coord_pairs)
    _coord = np.around(_coord, 5)  # correct for numerical precision
    _coord -= np.amin(_coord, axis=0)

    nrows, ncols = (max_y - min_y + 1,
                    max_x - min_x + 1)

    pixel_indices = _coord[:, 1] * ncols + _coord[:, 0]
    return pixel_indices.astype(np.int32), nrows, ncols


