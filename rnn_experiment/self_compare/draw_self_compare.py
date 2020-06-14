import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_FILE_PATH = "all_results.pkl"


def draw_all_pairs(df):
    if isinstance(df, str):
        df = pickle.load(open(df, "rb"))
    names = [n.replace("_result", "") for n in df.columns if n.endswith('_result')]
    for i, name_1 in enumerate(names):
        for name_2 in names[i + 1:]:
            draw_all_columns(df, name_1, name_2)


def draw_all_columns(df, name_1=None, name_2=None, draw_errors=True):
    interesting_columns = ['queries', 'time'] #, 'invariant_queries', 'property_queries'] #, 'time_log']
                           # + ['invariant_queries', 'property_queries', 'invariant_times_mean', 'property_times_mean']
    df[name_1 + '_invariant_times_mean'] = df[name_1 + '_invariant_times'].apply(np.mean)
    df[name_1 + '_property_times_mean'] = df[name_1 + '_property_times'].apply(np.mean)
    df[name_2 + '_invariant_times_mean'] = df[name_2 + '_invariant_times'].apply(np.mean)
    df[name_2 + '_property_times_mean'] = df[name_2 + '_property_times'].apply(np.mean)
    df[name_1 + '_time_log'] = df[name_1 + '_time'].apply(np.log10)
    df[name_2 + '_time_log'] = df[name_2 + '_time'].apply(np.log10)
    for c in interesting_columns:
        draw_from_dataframe(df, name_1, name_2, draw_errors, c)


def draw_queries_from_df(df, name_1=None, name_2=None, draw_errors=True):
    draw_from_dataframe(df, name_1, name_2, draw_errors, 'queries')


def draw_time_from_df(df, name_1=None, name_2=None, draw_errors=True):
    draw_from_dataframe(df, name_1, name_2, draw_errors, 'time')


def draw_from_dataframe(df, name_1=None, name_2=None, draw_errors=True, draw_param='queries'):
    '''
    :param df: with columns: exp_name and foreach algorithm have: name_result, name_queries
    :param name_1: Optional, x axis, if None use an algorithm from df.columns
    :param name_2: Optional, y axis, if None use an algorithm from df.columns
    :return:
    '''
    if isinstance(df, str):
        df = pickle.load(open(df, "rb"))

    names = [n.replace("_result", "") for n in df.columns if n.endswith('_result')]

    if len(names) < 2:
        raise ValueError('data frame is invalid')
    if name_1 is not None and name_2 is not None:
        x_alg = name_1
        y_alg = name_2
        assert df.loc[:, x_alg + "_result"] is not None
        assert df.loc[:, y_alg + "_result"] is not None
    else:
        x_alg = names[0]
        y_alg = names[1]

    x_name = x_alg + "_" + draw_param
    y_name = y_alg + "_" + draw_param
    df[x_name] = df[x_name].astype('float')
    df[y_name] = df[y_name].astype('float')
    # Filter only to rows both algorithms proved

    df_filter = df.loc[(df[x_alg + '_result']) & (df[y_alg + '_result'])]
    if df_filter.shape[0] == 0:
        return
    max_no_error = max(df_filter[x_name].max(), df_filter[y_name].max())
    error_value = max_no_error + (max_no_error * 0.1)
    valid_border = max_no_error + (max_no_error * 0.05)
    # valid_border = max_no_error + 150
    if draw_errors:
        # max_val = max(df.max()[[x_name, y_name]])
        df_filter = df
        # error_value = max_no_error + 200
        df_filter.loc[df_filter[x_alg + "_result"] == False, x_name] = error_value  # max_no_error + 150
        df_filter.loc[df_filter[y_alg + "_result"] == False, y_name] = error_value  # max_no_error + 150

    categories = np.unique(df['net_name'])
    colors = ['#FF0000', '#0000FF', '#333333', '#FF00FF']
    colordict = dict(zip(categories, colors))

    df_filter["color"] = df_filter['net_name'].apply(lambda x: colordict[x])
    # ax.scatter(df[xcol], df[ycol], c=df.Color)

    # df_filter.plot.scatter(x=x_name, y=y_name, c=df_filter.color, label=df_filter['net_name'])
    # df_filter.plot.scatter(x=x_name, y=y_name, color=df_filter.color)
    # scatter = plt.scatter(df_filter[x_name], y=df_filter[y_name], c=df_filter.color) #, cmap=colours)
    # plt.legend(*scatter.legend_elements(), labels=df_filter.net_name)
    #
    #

    sns.scatterplot(x=df_filter[x_name], y=df_filter[y_name], hue=df_filter.net_name)
    # plt.show()

    # print(df_filter)
    rgb = [i / 256 for i in (145, 40, 230)]
    plt.xlim(0, error_value + (error_value * 0.01))  # max_no_error + 200)
    plt.ylim(0, error_value + (error_value * 0.01))  # max_no_error + 200)
    plt.plot(plt.xlim(), plt.ylim(), '--', color=rgb + [0.6])
    if draw_errors:
        # valid_border = max_no_error + 50
        plt.hlines(valid_border, 0, valid_border, linestyles='dashed', color='orange')
        plt.vlines(valid_border, 0, valid_border, linestyles='dashed', color='orange')

    plt.title(draw_param)
    # plt.xlabel(x_alg.replace('_big', ''))
    # plt.ylabel(y_alg.replace('_big', ''))
    save_name = x_name + y_name  # + str(datetime.now()).replace('.', '') + ".png"
    save_name += ".png"
    # plt.savefig(save_name)
    plt.show()


if __name__ == "__main__":
    # # draw_all_pairs(df)
    # weighted_exp_summary = "results_model_20classes_rnn4_fc32_epochs40weighted_relative_weighted_big_absolute_weighted_absolute.pkl"
    # weighted_exp_summary  = "results_model_20classes_rnn4_fc32_epochs40.h5_randomexp_random_relative_weighted_relative.pkl"
    # df_weighted = pickle.load(open(weighted_exp_summary, "rb"))
    # draw_queries_from_df(df_weighted)
    # draw_time_from_df(df_weighted)

    # Random vs Weighted with relative

    path = os.path.join("pickles",
                        "results_model_20classes_rnn4_fc32_epochs40.h5_randomexp_random_relative_weighted_relative_iterate_absolute_weighted_absolute_random_big_absolute_weighted_big_absolute2020-01-15 19:37:29378275.pkl")

    path = "random_vs_weighted_realtive.pkl"
    path = os.path.join("pickles", "random_vs_weighted_relative.pkl")
    # path = os.path.join("pickles", "absolute_compars.pkl")
    # path = os.path.join("pickles", "inverse_vs_weighted_absolute.pkl")
    path = os.path.join("pickles", "relative_sigmoid_vs_weighted.pkl")
    path = os.path.join("pickles", "relative_absolute_all.pkl")


    dir_path = 'gurobi_random_compare'
    draw_all_pairs('gurobi_random_compare/combined.pkl')
    # for p in os.listdir(dir_path):
    #     file_p = os.path.join(dir_path, p)
    #     if os.path.isfile(file_p):
    #         draw_all_pairs(file_p)
    # folder_name = "pickles/exp210120/combined"
    # draw_all_pairs("/home/yuval/projects/Marabou/pickles/new_pickles/combined/0.1gurobi_random_relative.pkl")
    # for p in os.listdir(folder_name):
    #     draw_all_pairs(os.path.join(folder_name, p))
    # draw_queries_from_df(pickle.load(open(path, 'rb'))) #, "random_big_absolute", "weighted_big_absolute")
    # draw_time_from_df(pickle.load(open(path, 'rb'))) #, "random_big_absolute", "weighted_big_absolute")
    # draw_from_dataframe(df_weighted, 'weighted_relative', 'weighted_big_absolute')

    # relative_exp_summary = "results_model_20classes_rnn4_fc32_epochs40random_relative_iterate_relative_weighted_relative.pkl"
    # df_relative = pickle.load(open(relative_exp_summary, "rb"))
    # draw_from_dataframe(df_relative, 'weighted_relative', 'random_relative')
    #
    # absolute_exp_summary = "results_model_20classes_rnn4_fc32_epochs40iterate_big_absolute_weighted_big_absolute_result.pkl"
    # df_absolute = pickle.load(open(absolute_exp_summary, "rb"))
    # draw_from_dataframe(df_absolute, 'iterate_big_absolute', 'weighted_big_absolute')
