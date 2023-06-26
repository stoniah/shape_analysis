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
    if modifier == 'F':
        column_names.append(f'{index_name}_ED')
        column_names.append(f'{index_name}_ES')
    return column_names


def main():
    parser = argparse.ArgumentParser(
        prog='lvfg',
        description=description
    )
    parser.add_argument('dir', type=str, help='directory with segmented left ventricle records')
    parser.add_argument('-i', '--interpolate', type=int, metavar='N', default=30,
                        help='perform time interpolation setting record frame count to N (default: N=30)')
    parser.add_argument('-m', '--max_interpolate', action='store_true',
                        help='set N for --interpolate flag as max record length in dir, overrides the --interpolate flag')
    parser.add_argument('-f', '--fourier', type=int, metavar='F', default=None,
                        help='perform time interpolation with F fourier harmonics to ensure continuity between last and first frames'
                        )
    parser.add_argument('-r', '--regions', type=int, metavar='R', default=20,
                        help='LV is subdivided into R regions for calculation of heterogeneity indexes (default is 20)')
    parser.add_argument('-o', '--out', type=str, help='filename where to save the table with indexes',
                        default='lvfg_table.csv')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    data = reading_utils.load_data(Path(args.dir), only_norm=True, only_endo=True)
    if args.max_interpolate:
        record_lengths = [len(record['coordinates']) for record in data]
        n_frames = max(*record_lengths)
    else:
        n_frames = args.interpolate

    n_regions = args.regions
    if args.fourier is not None:
        n_harmonics = args.fourier
        fourier_interpolate_kwargs = {'n_harmonics': n_harmonics, "n_points_out": None}
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

    record_features = {
        'SHI': shape_features.get_spacial_heterogeneity_index,
        'THI': shape_features.get_temporal_heterogeneity_index
    }
    frame_features = {
        'SI': shape_features.get_sphericity_index,
        'GSI': shape_features.get_gibson_sphericity_index,
        'CI': shape_features.get_conicity_index,
        'EI': shape_features.get_eccentricity_index,
        'FSPI': shape_features.get_fourier_shape_power_index,
    }
    region_features = {
        'ES': shape_features.get_local_ES,
        'AS': shape_features.get_local_asynchronism_indexes,
        'EF': shape_features.get_local_EF
    }

    table_rows = []
    # column_names = [
    #     'SHI', 'THI',
    #     *generate_column_names_for_each_frame('SI', n_frames),
    #     *generate_column_names_for_each_frame('GSI', n_frames),
    #     *generate_column_names_for_each_frame('CI', n_frames),
    #     *generate_column_names_for_each_frame('EI', n_frames),
    #     *generate_column_names_for_each_frame('FSPI', n_frames),
    #     *generate_column_names_for_each_frame('ES', n_regions, 'R'),
    #     *generate_column_names_for_each_frame('AS', n_regions, 'R'),
    #     *generate_column_names_for_each_frame('EF', n_regions, 'R')
    # ]
    column_names = []
    for record_feature_name in record_features:
        column_names.append(record_feature_name)
    for frame_feature_name in frame_features:
        column_names.extend(generate_column_names_for_each_frame(frame_feature_name, n_frames))
    for region_feature_name in region_features:
        column_names.extend(generate_column_names_for_each_frame(region_feature_name, n_regions, 'R'))

    record_ids = []
    for contour in tqdm.tqdm(contours):
        record_id = f"{contour['patient_code']}_{contour['operation_status']}_{contour['wall_type']}"
        record_ids.append(record_id)
        contour_points_time = contour['coordinates']

        table_row = []
        for record_feature_name, record_feature in record_features.items():
            if record_feature_name == 'THI':
                record_feature_value = record_feature(
                    contour_points_time,
                    end_systolic_id=contour['end_systolic_id'])
            else:
                record_feature_value = record_feature(contour_points_time)
            table_row.append(record_feature_value)

        for frame_feature_name, frame_feature in frame_features.items():
            frame_feature_values = calculate_index_for_each_frame(frame_feature, contour_points_time)
            if frame_feature_name == 'FSPI':
                fspi_scaled = np.round(np.array(frame_feature_values).astype('float64') * 100, 1)
                frame_feature_values_final = fspi_scaled
            else:
                frame_feature_values_final = frame_feature_values
            table_row.extend(frame_feature_values_final)
            table_row.append(frame_feature_values_final[0])
            table_row.append(frame_feature_values_final[contour['end_systolic_id']])

        for region_feature_name, region_feature in region_features.items():
            if region_feature_name == 'AS':
                region_feature_values = region_feature(contour_points_time, contour['end_systolic_id'], n_regions)
            else:
                region_feature_values = region_feature(contour_points_time, n_regions)
            table_row.extend(region_feature_values)

        table_rows.append(table_row)

    df = pd.DataFrame(table_rows, columns=column_names, index=record_ids)
    df.index.name = 'record_id'
    df.to_csv(args.out, float_format='%3.3f')


if __name__ == '__main__':
    main()
