from preprocess import *
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from gower import gower_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def detect_geo(df):
    dbscan = DBSCAN(eps=.35, min_samples=2)
    dbscan.fit(df)
    df['cluster'] = pd.Series(dbscan.labels_).astype(str)
    return df


def save_fig(df, title=None):
    tsne = TSNE(perplexity=40, early_exaggeration=10).fit_transform(df.drop(columns=['cluster']))
    plt.figure()
    sns.scatterplot(tsne[:,0], tsne[:,1], hue=df['cluster'])
    if title:
        plt.title(title)
        plt.savefig(title)
    else:
        plt.savefig('temp.png')


def detect_cat(dfs):
    errors = []
    for name in dfs.keys():
        df = dfs[name]['df']
        mat = gower_matrix(df)
        dbscan = DBSCAN(eps=.30, min_samples=2, metric='precomputed')
        dbscan.fit(mat)
        df['cluster'] = dbscan.labels_
        idx = df.index[df['cluster'] != 0].tolist()
        print(df['cluster'])
        errors += idx
    return errors


def main():
    try:
        with fileinput.input() as f:
            df = to_df(f)
    except AttributeError:
        with open('raw/rohbau.json') as f:
            df = to_df(f)
    df, ID = drop_uniques_consts(df)
    df_geo = add_geometry(df)
    geometric_columns = get_geometric_columns(df_geo, min_corr=.5)
    df_geo = df_geo[geometric_columns]
    df_geo = df_geo.fillna(0)
    df_geo = pd.DataFrame(MinMaxScaler().fit_transform(df_geo), columns=df_geo.columns)
    result_geo = detect_geo(df_geo)
    save_fig(result_geo, 'Geo clusters')
    

    
    grouper = get_categorical_groupers(df)
    groups = get_groups(df, grouper[0], min_samples=7)
    groups = convert_to_float(groups)
    groups = one_hot_encoding(groups)
    groups = min_max_scaler(groups)
    errors = detect_cat(groups)

if __name__ == '__main__':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        main()
