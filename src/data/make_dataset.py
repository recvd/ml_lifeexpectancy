# -*- coding: utf-8 -*-
import click
import logging
import json
import pandas as pd
from pathlib import Path

# from dotenv import find_dotenv, load_dotenv

# @click.command()
# @click.argument('raw_filepath', type=click.Path())
# @click.argument('interim_filepath', type=click.Path())
# @click.argument('processed_filepath', type=click.Path())
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

    #############################################################
    ################ Life Expectancy Outcome ####################
    #############################################################

    le_birth = pd.read_csv(raw_filepath / 'US_A.csv',
                         usecols=['Tract ID', 'e(0)'],
                         dtype={'Tract ID': "object"}) \
        .rename(columns={'Tract ID': 't10_cen_uid_u_2010'}) \
        .set_index('t10_cen_uid_u_2010')

    le_other = pd.read_csv(raw_filepath / 'US_B.csv',
                           usecols=['Tract ID', 'Age Group', 'e(x)'],
                           dtype={'Tract ID': "object"}) \
        .rename(columns={'Tract ID': 't10_cen_uid_u_2010'}) \
        .set_index(['t10_cen_uid_u_2010', 'Age Group']) \
        .sort_index() \
        .loc[(slice(None), ['15-24', '35-44', '55-64']), :] \
        .unstack() \
        .reindex(le_birth.index)  # use the same tracts for all experiments

    le_other.columns = ['e(20)', 'e(40)', 'e(60)']

    # le_birth.to_csv(processed_filepath / 'y_00.csv', header=True)
    # le_other['e(20)'].to_csv(processed_filepath / 'y_20.csv', header=True)
    # le_other['e(40)'].to_csv(processed_filepath / 'y_40.csv', header=True)
    # le_other['e(60)'].to_csv(processed_filepath / 'y_60.csv', header=True)


    ##############################################################
    ################## Priority Dataset ##########################
    ##############################################################

    with open(raw_filepath / 'T10_Priority_Wide_Interpolated.csv', 'r') as f:
        cols = f.readline().strip().split(',')

    proj_cols = [x for x in cols if x[-4:] in years]# and
    # get all the priority NETS columns for later
    net_cols = ['t10_cen_uid_u_2010'] + [x[:11] + '_d_' + x[14:] for x in cols if '_net_' in x]

    data_X = pd.read_csv(raw_filepath / 'T10_Priority_Wide_Interpolated.csv', usecols=proj_cols,
                         dtype={'t10_cen_uid_u_2010': "object"}) \
        .set_index('t10_cen_uid_u_2010')

    # Create % younger than 25 (this method is far less than ideal)
    ag25up = data_X.filter(regex='.*(_pop_c_|ag25up).*')
    ag25up_coltuples = [(x[:-4], x[-4:]) for x in ag25up.columns]
    ag25up.columns = pd.MultiIndex.from_tuples(ag25up_coltuples)
    ag25up_long = ag25up.stack()
    ag25dwn_p = ((ag25up_long['t10_ldb_pop_c_'] - ag25up_long['t10_ldb_ag25up_c_'])
           / ag25up_long['t10_ldb_pop_c_']).unstack()
    ag25dwn_p.columns = ['t10_ldb_ag25dwn_p_' + x for x in ag25dwn_p.columns]

    # Create % older than 65
    ag65up = data_X.filter(regex='.*(_pop_c_|a60up).*')
    ag65up_coltuples = [(x[:-4], x[-4:]) for x in ag65up.columns]
    ag65up.columns = pd.MultiIndex.from_tuples(ag65up_coltuples)
    ag65up_long = ag65up.stack()
    ag65up_p = (ag65up_long['t10_ldb_a60up_c_'] / ag65up_long['t10_ldb_pop_c_']) \
        .unstack()
    ag65up_p.columns = ['t10_ldb_ag60up_p_' + x for x in ag65up_p.columns]

    # Add our new measure
    data_X = pd.concat([data_X, ag25dwn_p, ag65up_p], axis=1)

    # Get rid of all count variables, including nets
    no_count_cols = [x for x in data_X.columns if '_c_' not in x]
    data_X = data_X[no_count_cols]


    drop_cols = ['t10_gis_area_l_2010',
                 'm10_cen_uid_u_2010',
                 'm10_cen_memi_x_2010',
                 'c10_cen_uid_u_2010',
                 'z10_cen_uid_u_2010']

    data_X = data_X.drop(columns=drop_cols) \
        .reindex(le_birth.index)

    data_X.columns = pd.Index([(x[:-5], int(x[-4:])) for x in data_X.columns])

    X_priority = data_X.groupby(axis=1, level=0).mean()
    X_priority.to_csv(interim_filepath / 'X_priority.csv')

    ###########################################################
    #################### NETS Dataset #########################
    ###########################################################

    X_nets_allyrs = pd.read_csv(raw_filepath / 'recvd_t10_vars_v8_20190607.csv', usecols=net_cols,
                         dtype={'t10_cen_uid_u_2010': "object"}) \
        .set_index('t10_cen_uid_u_2010') \
        .reindex(le_birth.index)

    X_nets_allyrs.columns = pd.Index([(x[:-5], int(x[-4:])) for x in X_nets_allyrs.columns])
    X_nets = X_nets_allyrs.groupby(axis=1, level=0).mean()
    X_nets.to_csv(interim_filepath / 'X_nets.csv')

    # Split predictive data by Variable Set
    X_all = pd.concat([X_priority, X_nets], axis=1) \
        .dropna(how='any')

    final_index = le_birth.index.intersection(X_all.index)
    X_all = X_all.reindex(final_index)
    le_birth = le_birth.reindex(final_index)
    le_other = le_other.reindex(final_index)

    le_birth.to_csv(processed_filepath / 'y_00.csv', header=True)
    le_other['e(20)'].to_csv(processed_filepath / 'y_20.csv', header=True)
    le_other['e(40)'].to_csv(processed_filepath / 'y_40.csv', header=True)
    le_other['e(60)'].to_csv(processed_filepath / 'y_60.csv', header=True)

    # Var Set 1
    p1_features = ['t10_ldb_hinci_m',
                   't10_ldb_pop_d',
                   't10_ldb_nhblk_p',
                   't10_ldb_hisp_p',
                   't10_ldb_col_p']
    X_p1 = X_all[p1_features]
    X_p1.to_csv(processed_filepath / 'X_varGroup1.csv')

    # Var Set 2
    p2_features = [
        "t10_ldb_hinci_m",
        "t10_ldb_pop_d",
        "t10_ldb_ag25dwn_p",
        "t10_ldb_ag60up_p",
        "t10_ldb_nhblk_p",
        "t10_ldb_hisp_p",
        "t10_ldb_col_p",
        "t10_ldb_lep_p",
        "t10_ldb_mrenti_m",
        "t10_ldb_multi_p",
        "t10_ldb_nhwht_p",
        "t10_ldb_asian_p",
        "t10_ldb_fb_p",
        "t10_ldb_hs_p",
        "t10_ldb_unemp_p",
        "t10_ldb_npov_p",
        "t10_ldb_vac_p",
        "t10_ldb_own_p",
        "t10_ldb_mhmvali_m"
    ]
    X_p2 = X_all[p2_features]
    X_p2.to_csv(processed_filepath / 'X_varGroup2.csv')

    # Var Set 3
    X_p3 = X_nets.reindex(final_index)
    X_p3.to_csv(processed_filepath / 'X_varGroup3.csv')

    # Var Set 4
    X_p4 = X_all
    X_p4.to_csv(processed_filepath / 'X_varGroup4.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    data_dir = Path.cwd() / 'data'

    raw_filepath = '../../data/raw'
    interim_filepath = '../../data/interim'
    processed_filepath = '../../data/processed'
    main(raw_filepath, interim_filepath, processed_filepath)
