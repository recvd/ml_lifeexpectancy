import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import Lasso

def lasso_model(X_train, y_train, param_grid, cv):
    """Functional code to perform all operations for a Linear Model and return a fitted GridSearchCV

    Keyword Arguments:
    estimator -- The scikit-learn estimator to use
    param_grid -- Dict object detailing parameters for the model
    cv -- the cross-validation generator object
    X_train -- training data features
    y_train -- training data outcome
    phase -- Which phase of analysis this is for labeling charts, as a string (Ex: 1)
    """

    pipe = Pipeline([
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('estimator', Lasso())
    ])

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='r2')
    grid.fit(X_train, y_train)

    return grid

def linear_lasso_model(X, y, cv, save_path):
    param_grid = {
        'estimator__alpha': np.logspace(-3, 3, 5),
        'poly__degree': [1]
    }

    fitted_lasso_grid = lasso_model(X, y, param_grid, cv)

    with open(save_path, 'wb') as f:
        pickle.dump(fitted_lasso_grid, f)

def poly_lasso_model(X, y, cv, save_path):
    param_grid = {
        'estimator__alpha': np.logspace(-3, 3, 5),
        'poly__degree': [1, 2]
    }

    fitted_lasso_grid = lasso_model(X, y, param_grid, cv)

    with open(save_path, 'wb') as f:
        pickle.dump(fitted_lasso_grid, f)


# Phase 1 models
if __name__ == "__main__":
    cv_5 = KFold(n_splits=5, shuffle=True, random_state=42)
    data_path = Path.cwd().parents[1] / 'data' / 'processed'
    model_dir = Path.cwd().parents[1] / 'models'
    data_files = [x.parts[-1] for x in data_path.glob('**/*')]

    y_files = [y for y in data_files if 'y_' in y]
    X_files = [x for x in data_files if 'X_' in x]
    for y_file in y_files:
        y = pd.read_csv(
            data_path / y_file,
            index_col=['t10_cen_uid_u_2010'],
            dtype={
                't10_cen_uid_u_2010': object
            },
            squeeze=True
                        )
        for X_file in X_files:
            X = pd.read_csv(
                data_path / X_file,
                index_col=['t10_cen_uid_u_2010'],
                dtype={
                    't10_cen_uid_u_2010': object
                }
                            )
            X.dropna(how='all', inplace=True)
            y = y.reindex_like(X)
            X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, random_state=42)

            print('Variable Group {}, Life Expectancy at {}'.format(X_file[-5], y_file[2:4]))
            linear_lasso_model(X_train, y_train, cv_5, model_dir / 'linearLasso_varGroup{}_{}.pkl'
                               .format(X_file[-5], y_file[2:4])
                               )
            poly_lasso_model(X_train, y_train, cv_5, model_dir / 'polyLasso_varGroup{}_{}.pkl'
                                .format(X_file[-5], y_file[2:4])
                               )

