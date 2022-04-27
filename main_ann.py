import os
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from weighting_methods import stat_variance_weighting
from normalizations import minmax_normalization
from rank_preferences import rank_preferences
from correlations import spearman

from saw import SAW


# Functions for result visualizations
def plot_scatter(data, model_compare):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        data: dataframe
        model_compare : list[list]

    Examples
    ----------
    >>> plot_scatter(data. model_compare)
    """

    #sns.set_style("darkgrid")
    list_rank = np.arange(1, len(data) + 2, 2)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 18, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 18)
        ax.set_ylabel(el[1], fontsize = 18)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        x_ticks = ax.xaxis.get_major_ticks()
        y_ticks = ax.yaxis.get_major_ticks()

        ax.set_xlim(-1.5, len(data) + 2)
        ax.set_ylim(0, len(data) + 2)

        ax.grid(True, linestyle = '--')
        ax.set_axisbelow(True)
    
        plt.tight_layout()
        plt.savefig('results/scatter_' + el[0] + '.pdf')
        plt.show()


def plot_rankings(results):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        results : dataframe
            Dataframe with columns containing real and predicted rankings.

    Examples
    ---------
    >>> plot_rankings(results)
    """

    model_compare = []
    names = list(results.columns)
    model_compare = [[names[0], names[1]]]
    results = results.sort_values('Real rank')
    #sns.set_style("darkgrid")
    plot_scatter(data = results, model_compare = model_compare)

def main():
    warnings.filterwarnings("ignore")

    # =================================================================
    
    # Part 1
    # Datasets preparation
    '''
    path = 'DATASET'
    m = 30

    str_years = [str(y) for y in range(2016, 2021)]
    list_alt_names = ['A' + str(i) for i in range(1, m + 1)]
    list_alt_names_latex = [r'$A_{' + str(i + 1) + '}$' for i in range(0, m)]
    preferences = pd.DataFrame(index = list_alt_names)
    rankings = pd.DataFrame(index = list_alt_names)


    for el, year in enumerate(str_years):
        file = 'data_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        df_data = data.iloc[:len(data) - 1, :]
        # types
        types = data.iloc[len(data) - 1, :].to_numpy()
        
        list_of_cols = list(df_data.columns)
        # matrix
        matrix = df_data.to_numpy()
        
        # weights
        weights = stat_variance_weighting(matrix)
        saw = SAW()
        pref = saw(matrix, weights, types)
        rank = rank_preferences(pref, reverse = True)
        rankings[year] = rank
        preferences[year] = pref

        # normalized matrix for ML dataset
        yl = [year] * df_data.shape[0]
        nmat = minmax_normalization(matrix, types)
        df_nmat = pd.DataFrame(data=nmat, index = list_alt_names, columns = list(data.columns))
        df_nmat['Year'] = yl
        df_nmat['Pref'] = pref
        if el == 0:
            df_nmat_full = copy.deepcopy(df_nmat)
        else:
            df_nmat_full = pd.concat([df_nmat_full, df_nmat], axis = 0)

    rankings = rankings.rename_axis('Ai')
    rankings.to_csv('results/rankings.csv')
    preferences = preferences.rename_axis('Ai')
    preferences.to_csv('results/preferences.csv')

    df_nmat_full = df_nmat_full.rename_axis('Ai')
    df_nmat_full.to_csv('results/df_nmat_full.csv')

    df_train = df_nmat_full[(df_nmat_full['Year'] != '2020')]
    df_test = df_nmat_full[df_nmat_full['Year'] == '2020']

    df_train.to_csv('results/dataset_train.csv')
    df_test.to_csv('results/dataset_test.csv')
    '''

    
    # Machine Learning procedures
    # Part 2
    # load the data
    df_train = pd.read_csv('results/dataset_train.csv', index_col = 'Ai')
    df_test = pd.read_csv('results/dataset_test.csv', index_col = 'Ai')

    df_train = df_train.drop('Year', axis = 1)
    df_test = df_test.drop('Year', axis = 1)

    # train dataset
    X_train = df_train.iloc[:, :-1].to_numpy()
    y_train = df_train.iloc[:, -1].to_numpy()
    

    # ======================================================================
    # Selection of hyperparameters for MLP Regressor Model using GridSearchCV
    '''
    # grid search cv
    mlp = MLPRegressor()

    parameter_space = {
        'hidden_layer_sizes': [(100, ), (200, ), (500, )],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.001, 0.0001, 0.00001],
        'learning_rate': ['constant','adaptive'],
        'max_iter': [200, 500, 1000]}

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=4)
    clf.fit(X_train, y_train)

    print('Best parameters found:\n', clf.best_params_)
    '''

    
    # Testing MLP Regressor model with parameters selected in previous step on test dataset
    m = 30
    list_alt_names_latex = [r'$A_{' + str(i + 1) + '}$' for i in range(0, m)]

    # test dataset
    X_test = df_test.iloc[:, :-1].to_numpy()
    y_test = df_test.iloc[:, -1].to_numpy()

    rankings_results = pd.DataFrame(index = list_alt_names_latex)

    model = MLPRegressor(hidden_layer_sizes = (500, ), 
    activation = 'tanh', 
    solver = 'lbfgs', 
    alpha = 0.0001, 
    learning_rate = 'constant', 
    learning_rate_init = 0.001, 
    max_iter=1000,
    tol = 0.0001, 
    shuffle = True,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_rank = rank_preferences(y_test, reverse = True)
    pred_rank = rank_preferences(y_pred, reverse = True)

    # Calculation of correlation between real and predicted rank
    print(spearman(test_rank, pred_rank))
    rankings_results['Real rank'] = test_rank
    rankings_results['Predicted rank'] = pred_rank
    rankings_results.to_csv('results/rankings_results.csv')

    plot_rankings(rankings_results)
    
    
    
if __name__ == '__main__':
    main()