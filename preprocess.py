import json
import pandas as pd
import numpy as np
import fileinput
from typing import TextIO


def to_df(file: TextIO):
    data = json.load(file)
    elements = data['result']['Elements']
    property_definitions = data['result']['ParameterDetails']
    structured = []
    for i in elements:
        row = {}
        bb = i['BoundingBox']
        for n, j in enumerate(bb['Axis']):
            axis_flat = zip(map(lambda x: f'Axis{n}'+ x, j.keys()), j.values())
            row.update(axis_flat)
        center = zip(map(lambda x: f'Center{x}', bb['Center'].keys()), bb['Center'].values())
        row.update(center)
        extent = zip(map(lambda x: f'Extent{x}', bb['Extent'].keys()), bb['Extent'].values())
        row.update(extent)
        row['WorldOrientation'] = bb['WorldOrientation']
        for k, v in property_definitions.items():
            row.update({v: i['Properties'].get(k, pd.NA)})
        structured.append(row)
    df = pd.DataFrame(structured)
    return df


def drop_uniques_consts(df: pd.DataFrame):
    cols = df.columns.copy()
    df = df.dropna(how='all', subset=[cols[x] for x in range(len(cols)) if x > 15])
    for c in cols:
        unique_string = df[c].nunique() == len(df[c].dropna())
        almost_unique_string = df[c].nunique() >= len(df[c].dropna()) * .95
        is_string = df[c].dtype == 'object'
        constant_or_novalue = len(df[c].value_counts()) <= 1
        if constant_or_novalue or (is_string and (unique_string or almost_unique_string)) or (is_string and np.all(df[c].str.contains('{'))):
            if unique_string and np.sum(df[c].isna()) == 0:
                ID = df[c].copy()
            df.drop(c, axis=1, inplace=True)
    return df, ID


def get_categorical_groupers(df, min_groups=25):
    n = df.copy()
    n = n.replace(pd.NA, np.NAN)
    n = n.astype('float64', errors='ignore')
    lst = []
    for c in n.columns:
        if n[c].dtype == 'object':
            vc = n[c].value_counts()
            if np.sum(n[c].isna()) <= 2 and (unique:=len(vc)) >= min_groups:
                r_std = vc.std()/vc.mean()
                if len(lst) != 0 and not any([x[1] == unique and x[2] == r_std for x in lst]):
                    lst.append((c, unique, r_std))
                elif len(lst) == 0:
                    lst.append((c, unique, r_std))
    lst = sorted(lst, key=lambda x: x[2])
    return [x[0] for x in lst]


def get_groups(df: pd.DataFrame, grouper: str, min_samples: int = 6):
    dfs = {}
    for n, g in df.groupby(grouper):
        if len(g) >= min_samples:
            for c, v in zip(g.columns, np.sum(g.isna())):
                if v == len(g) or g[c].nunique() == 1:
                    g.drop(c, axis=1, inplace=True)
            dfs[n] = {'length': len(g), 'cols': len(g.columns), 'df': g}
    return dfs


def convert_to_float(dfs):
    for v in dfs.values():
        for c in v['df'].columns:
            if v['df'][c].dtype == 'object':
                if np.all(v['df'][c].dropna(axis=0).str.replace('.', '').str.replace('-', '').str.isnumeric()):
                    v['df'][c] = v['df'][c].astype('float64', errors='ignore')
    return dfs

def min_max_scaler(dfs: dict):
    for v in dfs.values():
        for c in v['df'].columns:
            if type(v['df'][c]) == pd.DataFrame:
                print(v['df'][c])
            if v['df'][c].dtype == 'float64':
                cl = v['df'][c]
                v['df'][c] = (cl - cl.min()) / (cl.max() - cl.min())
    return dfs

def one_hot_encoding(dfs: dict):
    for v in dfs.values():
        concat = []
        for c in v['df'].columns:
            if (n:=v['df'][c]).dtype == 'object':
                dummies = pd.get_dummies(n)
                concat.append(dummies)
                v['df'].drop(c, axis=1, inplace=True)
        v['df'] = pd.concat([v['df']] + concat, axis=1)
        v['df'] = v['df'].loc[:,~v['df'].columns.duplicated()]
    return dfs


def view_groups(dfs: dict):
    count = 0
    for k, v in dfs.items():
        print(k)
        print(f'cols count: {v["cols"]}, length of dataframe: {len(v["df"])}', end='\n\n')
        count += len(v['df'])
    print('sample loss: ', len(df) - count)


def add_geometry(df):
    """
    calculate geometic features from ExtentXYZ 
    """
    # copying the original df is necessary because we want to return a copy and not modify the original
    n = df.copy()
    n['Volume'] = n['ExtentX'] * n['ExtentY'] * n['ExtentZ']
    n['AreaXY'] = n['ExtentX'] * n['ExtentY']
    n['AreaYZ'] = n['ExtentZ'] * n['ExtentY']
    n['AreaXZ'] = n['ExtentZ'] * n['ExtentX']
    n = n.replace(pd.NA, np.NAN)
    n = n.astype('float64', errors='ignore')
    drop = [x for x in n.columns if n[x].dtype == 'object']
    return n.drop(columns=drop)
    

def get_geometric_columns(df: pd.DataFrame, min_corr: float=.25):
    """
    finds the columns that are correlated with geometry columns
        parameters:
            df: DataFrame, Positional
                data in dataframe
            min_corr: float, Optional
                minimum correlation with geometry columns to count the column as geometric
        return: DataFrame
    """
    geometry = ['Volume', 'AreaXY', 'AreaXZ',
            'AreaYZ', 'ExtentX', 'ExtentY',
            'ExtentZ', 'Axis0X', 'Axis0Y',
            'Axis0Z', 'Axis1X', 'Axis1Y',
            'Axis1Z', 'Axis2X', 'Axis2Y', 'Axis2Z']

    corr = df.loc[:,df.dtypes=='float64'].corr()[geometry]
    return list(corr[~corr.index.isin(geometry)][corr > min_corr].dropna(axis=0, how='all').index)




    
if __name__ == '__main__':
    with pd.option_context('display.width', 1000, 'display.max_rows', None):
        try:
            with fileinput.input() as f:
                df = to_df(f)
        except AttributeError:
            with open('raw/rohbau.json') as f:
                df = to_df(f)
        df = drop_uniques_consts(df)
        df_geo = add_geometry(df)
        geometric_columns = get_geometric_columns(df_geo)
        groupers = get_categorical_groupers(df)[0]
        #view_groups(get_groups(groupers[0]))
