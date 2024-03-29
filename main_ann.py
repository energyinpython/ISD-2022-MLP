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

from saw import SAW

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from pyrepo_mcda import correlations as corrs
from sklearn.model_selection import cross_val_score


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



def plot_barplot(df_plot, legend_title):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.
    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.
        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).
    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """
    step = 2
    list_rank = np.arange(1, len(df_plot) + 1, step)
    # colors = ['#1f77b4', 'orange', 'green']

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 12)

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./results/' + 'bar_chart_' + legend_title + '.pdf')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (6, 4))
    sns.set(font_scale = 1.6)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="GnBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Rankings')
    plt.title(title + ' correlation coefficient')
    plt.tight_layout()
    plt.savefig('./results/' + 'correlations_' + title + '.pdf')
    plt.show()


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


def main():
    warnings.filterwarnings("ignore")

    # =================================================================
    
    # Part 1
    # Datasets preparation
    '''
    # Folder `DATA` includes performance values from the Eurostat database collected for selected 30 countries from 2010-2020.
    path = 'DATA'
    m = 30

    str_years = [str(y) for y in range(2010, 2021)]
    list_alt_names = ['A' + str(i) for i in range(1, m + 1)]
    list_alt_names_latex = [r'$A_{' + str(i + 1) + '}$' for i in range(0, m)]
    preferences = pd.DataFrame(index = list_alt_names)
    rankings = pd.DataFrame(index = list_alt_names)

    # types
    types = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1])


    for el, year in enumerate(str_years):
        file = 'data_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        list_of_cols = list(data.columns)
        # matrix
        matrix = data.to_numpy()
        print(types.shape)
        print(matrix.shape)
        
        # weights
        weights = stat_variance_weighting(matrix)
        saw = SAW()
        pref = saw(matrix, weights, types)
        rank = rank_preferences(pref, reverse = True)
        rankings[year] = rank
        preferences[year] = pref

        # normalized matrix for ML dataset
        yl = [year] * data.shape[0]
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

    # Folder `DATASET` includes dataset.csv, which contains a dataset that can be split into
    # training and test dataset. It contains features (normalized performance values) and
    # target variable (SAW utility function values)
    df_train.to_csv('DATASET/dataset.csv')
    df_test.to_csv('DATASET/dataset_test.csv')
    '''

    
    # Machine Learning procedures
    # Part 2
    # load the data
    df_dataset = pd.read_csv('DATASET/dataset.csv', index_col = 'Ai')
    df_dataset = df_dataset.drop('Year', axis = 1)
    
    dataset = df_dataset.to_numpy()
    X = dataset[:, :-1]
    y = dataset[:, -1]

    # Split the dataset into the train and test datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Shape of train dataset: ', X_train.shape)
    print('Shape of test dataset: ', X_test.shape)
    
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
    
    model = MLPRegressor(hidden_layer_sizes = (500, ), 
    activation = 'tanh', 
    solver = 'lbfgs', 
    alpha = 0.0001, 
    learning_rate = 'adaptive', 
    learning_rate_init = 0.001, 
    max_iter=1000,
    tol = 0.0001, 
    shuffle = True,
    )

    # Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 
    # 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 
    # 'recall', 'roc_auc']
    score = cross_val_score(model, X, y, cv=5, scoring = 'r2')
    df_score = pd.DataFrame(score)
    df_score.to_csv('results/cross_val_score.csv')
    
    ss = np.zeros((4, 2))
    df_scores = pd.DataFrame(ss, index = ['60 samples MLP', '60 samples OLS','30 samples MLP', '30 samples OLS'], columns = ['Spearman', 'r2'])

    # train MLP
    model.fit(X_train, y_train)
    # only for adam solver
    # pd.DataFrame(model.loss_curve_).plot()
    # plt.show()

    # MLP Sperman 60
    y_pred = model.predict(X_test)
    spearman_coeff = corrs.spearman(y_test, y_pred)
    print(spearman_coeff)
    df_scores.iloc[0, 0] = spearman_coeff

    # MLP r2 60
    r2_coeff = r2_score(y_test, y_pred)
    print(r2_coeff)
    df_scores.iloc[0, 1] = r2_coeff


    # Model OLS
    # https://www.statsmodels.org/dev/examples/notebooks/generated/predict.html

    # https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
    
    # https://www4.eco.unicamp.br/docentes/gori/images/arquivos/PanelData/HO235_Lesson4_PooledPanelData.pdf
    X_train_ols = sm.add_constant(X_train)
    X_test_ols = sm.add_constant(X_test)
    
    # train OLS
    model_ols = sm.OLS(y_train, X_train_ols)
    model_ols_res = model_ols.fit()

    y_pred_ols = model_ols_res.predict(X_test_ols)

    # OLS Spearman 60
    spearman_coeff = corrs.spearman(y_test, y_pred_ols)
    print(spearman_coeff)
    df_scores.iloc[1, 0] = spearman_coeff

    # OLS r2 60
    r2_coeff = r2_score(y_test, y_pred_ols)
    print(r2_coeff)
    df_scores.iloc[1, 1] = r2_coeff
    
    
    # plot
    x1 = np.arange(1, len(y_pred_ols) + 1, 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x1, y_test, "o", label = 'Real value')
    ax.plot(x1, y_pred, 'r-', linewidth = 4, label = "MLP prediction")
    ax.plot(x1, y_pred_ols, 'k--', linewidth = 2, label = "OLS prediction")
    # ax.set_xticks(x1)
    # ax.set_xticklabels(list_alt_names_latex, fontsize = 12)
    ax.tick_params(axis = 'both', labelsize = 14)
    ax.set_xlabel('Alternatives', fontsize = 14)
    ax.set_ylabel('Utility function value', fontsize = 14)
    # plt.legend(fontsize = 12)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=3, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 14)
    plt.grid(True, linestyle = ':')
    plt.tight_layout()
    plt.savefig('results/scatter_line_full.pdf')
    plt.show()
    

    
    # study for 2020 test dataset

    X_train = dataset[:, :-1]
    y_train = dataset[:, -1]

    df_test = pd.read_csv('DATASET/dataset_test.csv', index_col = 'Ai')
    df_test = df_test.drop('Year', axis = 1)
    
    dataset_test = df_test.to_numpy()
    X_test = dataset_test[:, :-1]
    y_test = dataset_test[:, -1]

    # train MLP
    model.fit(X_train, y_train)

    # MLP Spearman 30
    y_pred = model.predict(X_test)
    spearman_coeff = corrs.spearman(y_test, y_pred)
    print(spearman_coeff)
    df_scores.iloc[2, 0] = spearman_coeff

    # MLP r2 30
    r2_coeff = r2_score(y_test, y_pred)
    print(r2_coeff)
    df_scores.iloc[2, 1] = r2_coeff

    X_train_ols = sm.add_constant(X_train)
    X_test_ols = sm.add_constant(X_test)
    
    # train OLS
    model_ols = sm.OLS(y_train, X_train_ols)
    model_ols_res = model_ols.fit()

    y_pred_ols = model_ols_res.predict(X_test_ols)

    # OLS Spearman 30
    spearman_coeff = corrs.spearman(y_test, y_pred_ols)
    print(spearman_coeff)
    df_scores.iloc[3, 0] = spearman_coeff

    # OLS r2 30
    r2_coeff = r2_score(y_test, y_pred_ols)
    print(r2_coeff)
    df_scores.iloc[3, 1] = r2_coeff

    # save results
    list_alt_names_latex = [r'$A_{' + str(i + 1) + '}$' for i in range(0, len(y_pred_ols))]
    test_rank = rank_preferences(y_test, reverse=True)
    pred_rank = rank_preferences(y_pred, reverse=True)
    pred_rank_ols = rank_preferences(y_pred_ols, reverse=True)

    df = pd.DataFrame(index = list_alt_names_latex, columns = ['Real', 'MLP', 'OLS'])
    df['Real'] = test_rank
    df['MLP'] = pred_rank
    df['OLS'] = pred_rank_ols
    df = df.rename_axis('Ai')
    df.to_csv('results/models_rankings.csv')

    plot_barplot(df, 'Rankings')
    df_scores.to_csv('results/df_scores.csv')
    
    data = pd.read_csv('results/models_rankings.csv', index_col='Ai')
    # data = copy.deepcopy(df)
    method_types = list(data.columns)
    dict_new_heatmap_rs = Create_dictionary()
    for el in method_types:
        dict_new_heatmap_rs.add(el, [])
    
    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rs[j].append(corrs.spearman(data[i], data[j]))
        
    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types
    
    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rs, 'Spearman')
    
   

    
if __name__ == '__main__':
    main()