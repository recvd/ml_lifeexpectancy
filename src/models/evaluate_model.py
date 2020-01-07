import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from eli5.sklearn import PermutationImportance

def adj_r2(r2, n, p):
    """
    Calculates the adjusted R^2 regression metric
    Params:
    r2: The unadjusted r2
    n: Number of data points
    p: number of features
    """
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2

if __name__ == "__main__":
    model_folder = Path.cwd().parents[1] / 'models'
    data_folder = Path.cwd().parents[1] / 'data' / 'processed'
    X_files = [x for x in data_folder.glob('**/*.csv') if 'X' in str(x)]

    # Get feature names
    colnames = dict()
    for filename in X_files:
        with open(filename, 'r') as f:
            cols = f.readline().strip().split(',')[1:]
            colnames[filename.name[-5]] = cols

    r2_dict = dict()
    for model_filepath in model_folder.glob('**/*.pkl'):
        model_filename = model_filepath.name
        print('Calculating feature importances for {}'.format(model_filename))
        with open(model_folder / model_filename, 'rb') as f:
            model = pickle.load(f)

        best_r2 = model.best_score_
        pipe = model.best_estimator_
        est = pipe.named_steps['estimator']
        n = pipe \
            .named_steps['scaler'] \
            .n_samples_seen_
        p = len(est.coef_)

        # Performance
        best_adj_r2 = adj_r2(best_r2, n, p)
        r2_dict[model_filename] = best_adj_r2

        # Permutation Importance
        X_data = pd.read_csv(data_folder / 'X_varGroup{}.csv'.format(model_filename[-8]),
                             index_col='t10_cen_uid_u_2010',
                             dtype={'t10_cen_uid_u_2010': object}
                             )
        y_data = pd.read_csv(data_folder / 'y_{}.csv'.format(model_filename[-6:-4]),
                             index_col='t10_cen_uid_u_2010',
                             dtype={'t10_cen_uid_u_2010': object},
                             squeeze=True
                             )
        perm = PermutationImportance(pipe, scoring='r2') \
            .fit(X_data.values, y_data.values, cv='prefit')

        perm_results = np.mean(np.array(perm.results_), axis=0)
        perm_df = pd.DataFrame({
            # 'feature': [x[8:] for x in colnames[model_filename[-8]]],
            'feature': X_data.columns.tolist(),
            'importance': perm_results
        }) \
            .sort_values('importance', ascending=False) \
            .set_index('feature')


        # Coefficients
        features_final = pipe.named_steps['poly'].get_feature_names(colnames[model_filename[-8]])
        coef_df = pd.DataFrame.from_dict({
            'feature': features_final,
            'coef': est.coef_
            },
        ).set_index('feature')

        # coef_filename = model_filename + '_coef_df'
        feature_df = coef_df.join(perm_df, how='outer')
        feature_df.to_csv(model_folder / 'results' / '{}_feature_df.csv'.format(model_filename[:-4]))

    r2_dict = pd.DataFrame.from_dict(r2_dict, orient='index', columns=['adjusted_r2'])
    r2_dict.to_csv(model_folder / 'results' / 'lasso_r2_values.csv')

