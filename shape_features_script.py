from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import tqdm

import reading_utils
import shape_features

description = 'Left Ventricle Functional Geometry. Extracts functional geometry indexes into a table'


# def calculate_index(index_func, args):
#     try:
#         index_func(*args)
#     except:

def calculate_index_for_each_frame(index, contour_points_time):
    return [index(contour_points) for contour_points in contour_points_time]


def generate_column_names_for_each_frame(index_name, n_frames, modifier='F'):
    column_names = []
    for frame_id in np.arange(n_frames):
        column_name = f'{index_name}_{modifier}{frame_id}'
        column_names.append(column_name)
    return column_names


def main():
    parser = argparse.ArgumentParser(
        prog='lvfg',
        description=description
    )
    parser.add_argument('dir', type=str, help='directory with segmented left ventricle records')
    parser.add_argument('--interpolate', type=int, metavar='N',
                        help='perform time interpolation setting record framecount to N (default: N is max record length in dir)')
    parser.add_argument('--fourier', type=int, metavar='F', default=None,
                        help='perform time interpolation with F fourier harmonics to ensure continuity between last and first frames'
                        )
    parser.add_argument('--regions', type=int, metavar='R', default=20,
                        help='LV is subdivided into R regions for calculation of heterogeneity indexes (default is 20)')
    parser.add_argument('--out', type=str, help='filename where to save the table with indexes',
                        default='lvfg_table.csv')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    data = reading_utils.load_data(Path(args.dir), only_norm=False, only_endo=True)
    if args.interpolate is None:
        record_lengths = [len(record['coordinates']) for record in data]
        n_frames = max(*record_lengths)
    else:
        n_frames = args.interpolate

    n_regions = args.regions
    if args.fourier is not None:
        n_harmonics = args.fourier
        fourier_interpolate_kwargs = {'n_harmonics': args.fourier, "n_points_out": None}
    else:
        fourier_interpolate_kwargs = None
    contours = reading_utils.prepare_contours(
        data,
        scale=True,
        center=True,
        interpolate_time_points=n_frames,
        fourier_interpolate_kwargs=fourier_interpolate_kwargs,
        close_contours=False,
    )
    table_rows = []
    column_names = ['SHI', 'THI',
                    *generate_column_names_for_each_frame('SI', n_frames),
                    *generate_column_names_for_each_frame('CI', n_frames),
                    *generate_column_names_for_each_frame('CI', n_frames),
                    *generate_column_names_for_each_frame('EI', n_frames),
                    *generate_column_names_for_each_frame('FSPI', n_frames),
                    *generate_column_names_for_each_frame('ES', n_regions, 'R'),
                    *generate_column_names_for_each_frame('AS', n_regions, 'R'),
                    *generate_column_names_for_each_frame('EF', n_regions, 'R')
                    ]
    for contour in tqdm.tqdm(contours):
        contour_points_time = contour['coordinates']
        shi = shape_features.get_spacial_heterogeneity_index(contour_points_time)
        thi = shape_features.get_temporal_heterogeneity_index(contour_points_time,
                                                              end_systolic_id=contour['end_systolic_id'])
        si = calculate_index_for_each_frame(shape_features.get_sphericity_index, contour_points_time)
        gi = calculate_index_for_each_frame(shape_features.get_gibson_sphericity_index, contour_points_time)
        ci = calculate_index_for_each_frame(shape_features.get_conicity_index, contour_points_time)
        ei = calculate_index_for_each_frame(shape_features.get_eccentricity_index, contour_points_time)
        fspi = calculate_index_for_each_frame(shape_features.get_fourier_shape_power_index, contour_points_time)
        es = shape_features.get_local_ES(contour_points_time, n_regions)
        asynchronism = shape_features.get_local_asynchronism_indexes(contour_points_time, contour['end_systolic_id'],
                                                                     n_regions)
        ef = shape_features.get_local_EF(contour_points_time, n_regions)

        table_row = [shi, thi, *si, *gi, *ci, *ei, *fspi, *es, *asynchronism, *ef]
        table_rows.append(table_row)

    df = pd.DataFrame(table_rows, columns=column_names)
    df.to_csv(args.out)


if __name__ == '__main__':
    main()
