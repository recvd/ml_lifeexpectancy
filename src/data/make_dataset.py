# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path

# from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('raw_filepath', type=click.Path())
@click.argument('interim_filepath', type=click.Path())
@click.argument('processed_filepath', type=click.Path())
def main(raw_filepath, interim_filepath, processed_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    raw_filepath = Path(raw_filepath)
    interim_filepath = Path(interim_filepath)
    processed_filepath = Path(processed_filepath)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    years = ['2010', '2011', '2012', '2013', '2014']

    ###
    # Priority Dataset
    ###

    with open(raw_filepath / 'T10_Priority_Wide_Interpolated.csv', 'r') as f:
        cols = f.readline().strip().split(',')

    proj_cols = [x for x in cols if x[-4:] in years]

    data_X = pd.read_csv(raw_filepath / 'T10_Priority_Wide_Interpolated.csv', usecols=proj_cols,
                         dtype={'t10_cen_uid_u_2010': "object"}) \
        .set_index('t10_cen_uid_u_2010')

    data_y = pd.read_csv(raw_filepath / 'US_A.csv',
                         usecols=['Tract ID', 'e(0)'],
                         dtype={'Tract ID': "object"}) \
        .rename(columns={'Tract ID': 't10_cen_uid_u_2010'}) \
        .set_index('t10_cen_uid_u_2010')

    data_allyrs = data_X.join(data_y, how='right')

    drop_cols = ['t10_gis_area_l_2010',
                 'm10_cen_uid_u_2010',
                 'c10_cen_uid_u_2010',
                 'z10_cen_uid_u_2010']

    data_allyrs.drop(columns=drop_cols, inplace=True)

    X_priority_allyrs = data_allyrs.iloc[:, :-1]

    X_priority_allyrs.to_csv(interim_filepath / 'X_priority_allyrs.csv')

    # no more processing necessary for our target so it goes directly to "processed"
    data_allyrs.iloc[:, -1].to_csv(processed_filepath / 'y_priority.csv', header=True)

    X_priority_allyrs.columns = pd.Index([(x[:-5], int(x[-4:])) for x in X_priority_allyrs.columns])

    X_priority = X_priority_allyrs.groupby(axis=1, level=0).mean()
    X_priority.to_csv(processed_filepath / 'X_priority.csv')


    ###
    # NETS Dataset
    ###

    with open(raw_filepath / 'recvd_t10_vars_v8_20190607.csv', 'r') as f:
        nets_cols = f.readline().strip().split(',')

    # only take 'h'ierarchal 'nets' 'd'ensity variables
    net_bool = lambda x: 'net' in x and x[-4:] in years and x[11] == 'h' and x[13] == 'd'
    nets_cols = ['t10_cen_uid_u_2010'] + [x for x in nets_cols if net_bool(x)]

    nets_X_allyrs = pd.read_csv(raw_filepath / 'recvd_t10_vars_v8_20190607.csv', usecols=nets_cols,
                         dtype={'t10_cen_uid_u_2010': "object"}) \
        .set_index('t10_cen_uid_u_2010') \
        .reindex(data_y.index)

    nets_X_allyrs.columns = pd.Index([(x[:-5], int(x[-4:])) for x in nets_X_allyrs.columns])
    nets_X = nets_X_allyrs.groupby(axis=1, level=0).mean()
    nets_X.to_csv(processed_filepath / 'X_nets.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    data_dir = Path.cwd() / 'data'

    main()
