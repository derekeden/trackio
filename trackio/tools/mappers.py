import json
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

from ..tools.utils import (read_pkl, 
                           save_pkl, 
                           collect_agent_pkls, 
                           flatten,
                           flatten_dict_unique)

def read_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def save_json(json_file, obj):
    with open(json_file, 'w') as f:
        json.dump(obj, f, indent=2)

def std_mapper(meta):
    mapper = {}
    for k in meta.keys():
        for v in meta[k]:
            v = str(v)
            mapper[v] = k
            mapper[v.lower()] = k
            mapper[v.upper()] = k
            mapper[v.title()] = k
            mapper[v.replace('_',' ')] = k
            mapper[v.replace('-',' ')] = k
            mapper[v.replace('-','_')] = k
            mapper[v.replace('_','-')] = k  
        mapper[k] = k 
    return mapper

#these are hardcoded
code_mapper_file = f'{os.path.dirname(__file__)}/../supporting/ais_code_mapper.csv'
aiscodes_file = f'{os.path.dirname(__file__)}/../supporting/AIS_Codes.csv'
column_mapper_file = f'{os.path.dirname(__file__)}/../supporting/column_mapper.csv'
status_mapper_file = f'{os.path.dirname(__file__)}/../supporting/ais_status_mapper.csv'

class mappers:
    def __init__(self):
        pass
    @property
    def ais(self):
        out = {'Status': std_mapper(pd.read_csv(status_mapper_file).set_index('Mapped')['Raw'].apply(eval).to_dict()),
               'AISCode': std_mapper(pd.read_csv(code_mapper_file).set_index('Mapped')['Raw'].apply(eval).to_dict()),
               'Type': pd.read_csv(aiscodes_file, usecols=['code','type']).set_index('code')['type'].to_dict(),
               'TypeStandard': pd.read_csv(aiscodes_file, usecols=['code','typestandard']).set_index('code')['typestandard'].to_dict()}
        out['Status']['__file__'] = status_mapper_file
        out['AISCode']['__file__'] =  code_mapper_file
        return out
    @property
    def columns(self):
        out = std_mapper(pd.read_csv(column_mapper_file).set_index('Mapped')['Raw'].apply(eval).to_dict())
        out['__file__'] = column_mapper_file
        return out
    def update(self, old, new):
        #check if it can be updated
        msg = 'This mapper is not meant to be dynamically updated'
        assert '__file__' in old.keys(), msg
        #get old
        old_file = old['__file__']
        old = pd.read_csv(old_file)
        old['Raw'] = old['Raw'].apply(eval)
        old.set_index('Mapped', inplace=True)
        #update 
        #drop any nan incase
        for k in new.copy().keys():
            if not isinstance(k, str) and np.isnan(k):
                new.pop(k)
            elif new[k] == '':
                new.pop(k)
            else:
                pass
        raw = new.keys()
        mapped = new.values()
        add = pd.DataFrame({'Raw':raw,'Mapped':mapped})
        add.set_index('Mapped', inplace=True)
        for i in range(len(add)):
            row = add.iloc[i]
            mapped = row.name
            raw = row['Raw']
            if raw not in old.loc[mapped,'Raw']:
                old.loc[mapped,'Raw'].append(row['Raw'])
        #rewrite
        old.reset_index(inplace=True)
        old.to_csv(old_file, index=False)
        return print(f"Updated mapper in {old_file}")

#initialize for __init__
_mappers = mappers()

def drop_agent_meta(inp,
                    out_pth,
                    file):
    #read agent
    agent = read_pkl(file)
    #loop over meta cols
    for i in inp:
        agent.meta.pop(i, None)
    #write it back out
    out_file = f'{out_pth}/{os.path.basename(file)}'
    save_pkl(out_file, agent)

def drop_agent_data(inp,
                    out_pth,
                    file):
    #read agent
    agent = read_pkl(file)
    #if split file
    if len(agent.tracks) > 0:
        for tid in agent.tracks.keys():
            for i in inp:
                t = agent.tracks[tid]
                if i in t.columns:
                    t.pop(i)
    else:
        for i in inp:
            a = agent._data
            if i in a.columns:
                a.pop(i)
    #write it back out
    out_file = f'{out_pth}/{os.path.basename(file)}'
    save_pkl(out_file, agent)
    
def map_agent_meta(inp, 
                   out, 
                   mapper, 
                   out_pth,
                   drop, 
                   fill,
                   file):
    #read agent
    agent = read_pkl(file)
    #loop over mappers
    for i,o in zip(inp,out):
        m = mapper[i]
        #add new meta value
        if i in agent.meta.keys():
            agent.meta[o] = m.get(agent.meta[i], fill)
            if drop:
                agent.meta.pop(i)
        else:
            agent.meta[o] = fill
    #write it back out
    out_file = f'{out_pth}/{os.path.basename(file)}'
    save_pkl(out_file, agent)
    
def map_agent_data(inp,
                   out,
                   mapper,
                   out_pth,
                   drop,
                   fill,
                   args):
    #split the args
    pkl_files, tracks = args
    ntracks = len(tracks)
    #if all data
    if ntracks == 0:
        #loop over files
        for pkl_file in pkl_files:
            #read file
            agent = read_pkl(pkl_file)
            #if a split file
            if len(agent.tracks) > 0:
                #loop over tracks
                for tid in agent.tracks.keys():
                    #get dataframe
                    tdf = agent.tracks[tid]
                    #loop over mappers
                    for i,o in zip(inp,out):
                        m = mapper[i]
                        #if input column not missing
                        if i in tdf.columns:
                            tdf[o] = tdf[i].apply(lambda x: m.get(x,fill)) 
                            if drop:
                                tdf.pop(i)
                        #if it is missing
                        else:
                            tdf[o] = [fill]*len(tdf)
                    #replace track
                    agent.tracks[tid] = tdf
            #unsplit file
            else:
                for i,o in zip(inp,out):
                    m = mapper[i]
                    if i in agent._data.columns:
                        agent._data[o] = agent._data[i].apply(lambda x: m.get(x,fill)) 
                        if drop:
                            agent._data.pop(i)
                    else:
                        agent._data[o] = [fill]*len(agent._data)
            #write it back out
            out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
            save_pkl(out_file, agent)
    else:
        #loop over files
        for pkl_file in pkl_files:
            refresh = False
            #read file
            agent = read_pkl(pkl_file)
            #loop over tracks
            for tid in agent.tracks.keys():
                if f"{agent.meta['Agent ID']}_{tid}" in tracks:
                    refresh = True
                    tdf = agent.tracks[tid]
                    for i,o in zip(inp,out):
                        m = mapper[i]
                        if i in tdf.columns:
                            tdf[o] = tdf[i].apply(lambda x: m.get(x,fill)) 
                            if drop:
                                tdf.pop(i)
                        else:
                            tdf[o] = [fill]*len(tdf)
                    agent.tracks[tid] = tdf
            if refresh:
                #write it back out
                out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
                save_pkl(out_file, agent)
                
def map_agent_data_to_codes(inp,
                            mapper,
                            out_pth,
                            drop,
                            fill,
                            args):
    #split the args
    pkl_files, tracks = args
    ntracks = len(tracks)
    #if all data
    if ntracks == 0:
        #loop over files
        for pkl_file in pkl_files:
            #read file
            agent = read_pkl(pkl_file)
            #if a split file
            if len(agent.tracks) > 0:
                for tid in agent.tracks.keys():
                    tdf = agent.tracks[tid]
                    for i in inp:
                        m = mapper[i]
                        if i in tdf.columns:
                            mapped = tdf[i].apply(lambda x: m.get(x,fill)).values #fill if missing 
                            for val in m.values():
                                tdf[f'Code{val}'] = (mapped == val)
                            if drop:
                                tdf.pop(i)
                        else:
                            mapped = np.array([fill]*len(tdf))
                            for val in m.values():
                                tdf[f'Code{val}'] = (mapped == val)
                    agent.tracks[tid] = tdf
            #unsplit file
            else:
                for i in inp:
                    m = mapper[i]
                    if i in agent._data.columns:
                        mapped = agent._data[i].apply(lambda x: m.get(x,fill)).values #-1 if missing 
                        for val in m.values():
                            agent._data[f'Code{val}'] = (mapped == val)
                        if drop:
                            agent._data.pop(i)
                    else:
                        mapped = np.array([fill]*len(tdf))
                        for val in m.values():
                            agent._data[f'Code{val}'] = (mapped == val)
            #write it back out
            out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
            save_pkl(out_file, agent)
    else:
        #loop over files
        for pkl_file in pkl_files:
            refresh = False
            #read file
            agent = read_pkl(pkl_file)
            #loop over tracks
            for tid in agent.tracks.keys():
                if f"{agent.meta['Agent ID']}_{tid}" in tracks:
                    refresh = True
                    tdf = agent.tracks[tid]
                    for i in inp:
                        m = mapper[i]
                        if i in tdf.columns:
                            mapped = tdf[i].apply(lambda x: m.get(x,fill)).values #fill if missing 
                            for val in m.values():
                                tdf[f'Code{val}'] = (mapped == val)
                            if drop:
                                tdf.pop(i)
                        else:
                            mapped = np.array([fill]*len(tdf))
                            for val in m.values():
                                tdf[f'Code{val}'] = (mapped == val)
                    agent.tracks[tid] = tdf
            if refresh:
                #write it back out
                out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
                save_pkl(out_file, agent)
                  
def make_meta_mapper(inp, pkl_file):
    #out dict
    out = {i:[] for i in inp}
    #read the file
    agent = read_pkl(pkl_file)
    for i in inp:
        out[i].append(agent.meta[i])
    return out
                            
def make_data_mapper(inp, args):
    #split the args
    pkl_files, tracks = args
    ntracks = len(tracks)
    #out dict
    out = {i:[] for i in inp}
    #if all data
    if ntracks == 0:
        #read the file
        agent = collect_agent_pkls(pkl_files)
        full_dat = pd.concat([agent.data, *agent.append_data])
        for i in inp:
            out[i].extend(np.unique(full_dat[i]).tolist())
    else:
        for pkl_file in pkl_files:
            agent = read_pkl(pkl_file)
            for tid in agent.tracks.keys():
                #if not one of the tracks to process
                if f"{agent.meta['Agent ID']}_{tid}" in tracks:
                    #get unique values for each data col
                    for i in inp:
                        out[i].extend(np.unique(agent.tracks[tid][i]).tolist())
    #reduce to unique only
    for i in out.keys():
        out[i] = np.unique(out[i]).tolist()
    return out

def _make_col_mapper(file):
    return pd.read_csv(file, nrows=0).columns.tolist()

def make_col_mapper(files, ncores=1, fill_mapper={}):
    #get list of unique columns
    with mp.Pool(ncores) as pool:
        cols = pool.map(_make_col_mapper, 
                        tqdm(files, 
                             total=len(files), 
                             desc='Making column mapper'))
    #flatten and get unique only
    cols = np.unique(flatten(cols))
    #match standard columns
    vals = list(map(lambda x: fill_mapper.get(x, ''), cols))
    #return mapper
    return dict(zip(cols, vals))

def map_columns(chunk, col_mapper):
    #make new column names
    new_cols = []
    for col in chunk.columns:
        new_cols.append(col_mapper.get(col, col))
    #set new column names
    chunk.columns = new_cols
    return chunk

def _make_raw_data_mapper(data_col, col_mapper, file):
    #get the columns
    cols = pd.read_csv(file, nrows=0).columns.tolist()
    mapped_cols = [col_mapper.get(c, c) for c in cols]
    #see which columns/attributes exist and to read
    read_icols = []
    read_attrs = []
    for attr in data_col:
        if attr in mapped_cols:
            read_icols.append(mapped_cols.index(attr))
            read_attrs.append(attr)
    #read the file only once
    usecols = [cols[i] for i in read_icols]
    df = pd.read_csv(file, 
                     usecols=usecols)[usecols] #to reorder them
    df.columns = read_attrs
    #format to unique dict
    out = {}
    for col in df.columns:
        out[col] = df[col].unique().tolist()
    #return
    return out

def make_raw_data_mapper(files,
                         col_mapper=_mappers.columns,
                         data_col=['Status','AISCode'],
                         fill_mapper={'Status': _mappers.ais['Status'],
                                      'AISCode': _mappers.ais['AISCode']},
                         ncores=4):
    #if only one column
    if isinstance(data_col, str):
        data_col = [data_col]
        fill_mapper = {data_col:fill_mapper}
    #get list of unique values for each attribute/column
    with mp.Pool(ncores) as pool:
        func = partial(_make_raw_data_mapper,
                       data_col,
                       col_mapper)
        _out = pool.map(func, 
                        tqdm(files, 
                             total=len(files), 
                             desc='QCing data columns'))
    #reformat into key:list dict
    out = flatten_dict_unique(_out)
    #convert list to raw:mapped dict
    for key in out.keys():
        fill = fill_mapper.get(key, {})
        raw = out[key]
        mapped = [fill.get(v,None) for v in raw]
        out[key] = dict(zip(raw,mapped))
    if len(out.keys()) == 1:
        return out[key]
    else:
        return out
        
