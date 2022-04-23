import numpy as np
import pandas as pd

from normalizations import *
from rank_preferences import rank_preferences
from correlations import spearman

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor


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

    sns.set_style("darkgrid")
    list_rank = np.arange(1, len(data) + 2, 4)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 12)
        ax.set_ylabel(el[1], fontsize = 12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        x_ticks = ax.xaxis.get_major_ticks()
        y_ticks = ax.yaxis.get_major_ticks()

        ax.set_xlim(-1, len(data) + 1)
        ax.set_ylim(0, len(data) + 1)

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
    sns.set_style("darkgrid")
    plot_scatter(data = results, model_compare = model_compare)

def main():

    #load datasets
    # train
    df_train = pd.read_csv('dataset_train.csv', index_col = 'Ai')
    # validation
    df_val = pd.read_csv('dataset_val.csv', index_col = 'Ai')
    # test
    df_test = pd.read_csv('dataset_test.csv', index_col = 'Ai')

    df_train = df_train.drop('Year', axis = 1)
    df_val = df_val.drop('Year', axis = 1)
    df_test = df_test.drop('Year', axis = 1)

    # Training dataset
    X_train = df_train.iloc[:, :-1].to_numpy()
    y_train = df_train.iloc[:, -1].to_numpy()

    # Validation dataset
    X_val = df_val.iloc[:, :-1].to_numpy()
    y_val = df_val.iloc[:, -1].to_numpy()


    X_train = df_train.iloc[:, :-1].to_numpy()
    y_train = df_train.iloc[:, -1].to_numpy()

    # Test dataset
    X_test = df_test.iloc[:, :-1].to_numpy()
    y_test = df_test.iloc[:, -1].to_numpy()

    # Initialization of the MLPRegressor model object
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

    # Fit the MLPRegressor model
    model.fit(X_train, y_train)
    # Predict preference values for test dataset using trained MLPRegressor model.
    y_pred = model.predict(X_test)

    # Generate real ranking based on real preference values for the test dataset.
    test_rank = rank_preferences(y_test, reverse = True)
    # Generate predicted ranking based on predicted preference values.
    pred_rank = rank_preferences(y_pred, reverse = True)
    # Calculate the Spearman Rank Correlation Coefficient value to determine the correlation between real and predicted rankings.
    print(spearman(test_rank, pred_rank))

    # Save the results in CSV file.
    alts = [r'$A_{' + str(i) + '}$' for i in range(1, len(test_rank) + 1)]

    results = pd.DataFrame(index = alts)
    results['Real rank'] = test_rank
    results['Predicted rank'] = pred_rank
    results = results.rename_axis(r'$A_{i}$')
    results.to_csv('results/results.csv')

    # Visualization of real and predicted rankings convergence.
    plot_rankings(results)
    
if __name__ == '__main__':
    main()