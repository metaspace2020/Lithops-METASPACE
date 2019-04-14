import pywren_ibm_cloud as pywren
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


def reduce_chunks(results):
    final_result = []
    for res_list in results:
        final_result.extend(res_list)
    return final_result


def parse_spectrum_line(s):
    ind_s, mzs_s, int_s = s.split('|')
    return (int(ind_s),
            np.fromstring(mzs_s, sep=' ').astype('float32'),
            np.fromstring(int_s, sep=' '))


def read_dataset_spectra(config, bucket_name, input_data):
    def reduce_chunks(results):
        final_result = []
        for res_list in results:
            final_result.extend(res_list)
        return final_result

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=512)
    iterdata = [[f'{bucket_name}/{input_data["ds"]}', parse_spectrum_line]]
    # NOTE: we need to be absolutely sure that using chunk_size doesn't split a line into separate chunks
    pw.map_reduce(parse_txt, iterdata, reduce_chunks, chunk_size=64*1024**2)
    spectra = pw.get_result()
    pw.clean()

    return spectra


def read_dataset_coords(config, bucket_name, input_data):
    def parse_spectrum_coord(s):
        sp_i, x, y = map(int, s.split(','))
        return (sp_i, x, y)

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=128)
    iterdata = [[f'{bucket_name}/{input_data["ds_coord"]}', parse_spectrum_coord]]
    pw.map(parse_txt, iterdata)
    spectra_coords = pw.get_result()
    pw.clean()

    return spectra_coords


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


