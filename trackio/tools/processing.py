################################################################################

from . import utils
from .mappers import map_columns
from .mappers import _mappers as mappers
from ..classes.Agent import Agent

import pandas as pd
import pickle as pkl
import os
import numpy as np
from inpoly import inpoly2
from shapely.geometry import Polygon
from functools import partial
import multiprocessing as mp

################################################################################

must_cols = ['Time','X','Y']

def group_points(groupby, 
               chunksize,
               out_pth,
               col_mapper,
               meta_cols,
               data_cols,
               data_mappers, 
               prefix,
               raw_file):
    #if it's a file
    if isinstance(raw_file, str):
        if (not raw_file.lower().endswith('.csv')):
            print(f'Raw file "{raw_file}" is not a csv, skipping...')
            return
        # get the reader, if csv then chunk for big ones, feather can't be chunked
        reader = pd.read_csv(raw_file, 
                             chunksize=chunksize, 
                             on_bad_lines='warn')
        #split raw ais pings to pkls by groupby
        for chunk in reader:
            _group_points(chunk, 
                          groupby,
                          out_pth,
                          col_mapper,
                          meta_cols,
                          data_cols,
                          data_mappers,
                          prefix)
    #if raw dataframe
    if isinstance(raw_file, pd.DataFrame):
        _group_points(raw_file,
                      groupby,
                      out_pth,
                      col_mapper,
                      meta_cols,
                      data_cols,
                      data_mappers,
                      prefix)

def _group_points(chunk, 
                  groupby, 
                  out_pth, 
                  col_mapper,
                  meta_cols,
                  data_cols,
                  data_mappers,
                  prefix):
    #get process ID
    pid = mp.current_process().name.replace('SpawnPoolWorker-', 'processor')
    #format the data columns
    dat = map_columns(chunk, col_mapper)
    #get a list of all columns necessary to keep
    if isinstance(groupby, str):
        all_cols = np.unique(meta_cols + data_cols + [groupby]).tolist()
        check_cols = must_cols+[groupby]
    else:
        all_cols = meta_cols.copy() + data_cols.copy()
        all_cols.extend(groupby)
        all_cols = np.unique(all_cols).tolist()
        check_cols = must_cols.copy()
        check_cols.extend(groupby)
    #get columns vailable from the lists passed
    available_cols = [c for c in dat.columns if c in all_cols]
    #make sure groupby, time, x, y all in there
    msg = f'Data files/dataframe must at minimum contain {must_cols} and {groupby} columns'
    assert all([col in available_cols for col in check_cols]), msg
    #filter the data
    dat = dat.filter(available_cols)
    #convert all strings to datetime
    dat['Time'] = pd.to_datetime(dat['Time'])
    #delete missing/nan critical columns
    dat = dat.dropna(subset=must_cols)
    #format any data columns that have mappers
    for col in dat.columns:
        if col in data_cols:
            if col in data_mappers.keys():
                mapper = data_mappers[col]
                dat[col] = dat[col].apply(lambda x: mapper.get(x,x))
    #group by groupby column, append all entries to giant lists - this is faster than .groupby??
    dat = dat.sort_values(by=groupby)
    dat['id'] = range(len(dat))
    ids = dat[[groupby,'id']].groupby(groupby)['id'].agg(list).values
    grouped = [dat.iloc[i] for i in ids]
    #loop through groups, append each to appropriate binary file
    for i in range(len(grouped)):
        #get the group of pings
        group = grouped[i]
        #get the groupby and agent id
        aid = f"{prefix}" + str(group[groupby].iloc[0])
        #make meta and data
        meta = group[meta_cols].iloc[0].to_dict()
        data = pd.DataFrame(group[data_cols].to_dict(orient='list'))
        agent = Agent(aid, meta, data)
        #append the pickle
        outfile = f"{out_pth}/{aid}_{pid}.points"
        utils.append_pkl(outfile, agent)

def split_tracks_spatiotemporal(time, 
                                distance, 
                                out_pth,
                                remove,
                                method,
                                args):
    #split the args
    pkl_files, tracks = args
    #get agent with collected unsplit points
    agent = utils.collect_agent_pkls(pkl_files)
    #split the points using threshold
    agent = agent.split_tracks_spatiotemporal(time=time, 
                                              distance=distance,
                                              tracks=tracks,
                                              method=method)
    #now rewrite the pickles
    new_name = f'{agent.meta["Agent ID"]}.tracks'
    # out_file = pkl_files[0].replace(os.path.basename(pkl_file), new_name)
    out_file = os.path.abspath(os.path.join(out_pth, new_name))
    #add file to meta
    agent.agent_meta['File'] = out_file
    # rewrite pickles
    with open(out_file, "wb") as f:
        pkl.dump(agent, f)
    #if removing unsplit data after splitting
    if remove:
        for pkl_file in pkl_files:
            #incase it rewrote the same file
            if os.path.abspath(pkl_file) != out_file:
                os.remove(pkl_file)

def split_tracks_by_data(data_col,
                         out_pth,
                         remove,
                         args):
    #split the args
    pkl_files, tracks = args
    #get agent with collected unsplit points
    agent = utils.collect_agent_pkls(pkl_files)
    #split the points using threshold
    agent = agent.split_tracks_by_data(data_col,
                                       tracks=tracks)
    #now rewrite the pickles
    new_name = f'{agent.meta["Agent ID"]}.tracks'
    # out_file = pkl_files[0].replace(os.path.basename(pkl_file), new_name)
    out_file = os.path.abspath(os.path.join(out_pth, new_name))
    #add file to meta
    agent.agent_meta['File'] = out_file
    # rewrite pickles
    with open(out_file, "wb") as f:
        pkl.dump(agent, f)
    #if removing unsplit data after splitting
    if remove:
        for pkl_file in pkl_files:
            #incase it rewrote the same file
            if os.path.abspath(pkl_file) != out_file:
                os.remove(pkl_file)

def split_tracks_kmeans(n_clusters, 
                        feature_cols, 
                        out_pth,
                        return_error,
                        remove,
                        optimal_method,
                        kwargs,
                        args):
    #split the args
    pkl_files, tracks = args
    #get agent with collected unsplit points
    agent = utils.collect_agent_pkls(pkl_files)
    #split the tracks using kmeans clustering
    agent = agent.split_tracks_kmeans(n_clusters=n_clusters,
                                      feature_cols=feature_cols,
                                      return_inertia=return_error,
                                      tracks=tracks,
                                      optimal_method=optimal_method,
                                      **kwargs)
    #split if necessary
    if return_error:
        agent, error = agent
    #now rewrite the pickles
    new_name = f'{agent.meta["Agent ID"]}.tracks'
    # out_file = pkl_files[0].replace(os.path.basename(pkl_file), new_name)
    out_file = os.path.abspath(os.path.join(out_pth, new_name))
    #add file to meta
    agent.agent_meta['File'] = out_file
    # rewrite pickles
    utils.save_pkl(out_file, agent)
    #if removing unsplit data after splitting
    if remove:
        for pkl_file in pkl_files:
            #incase it rewrote the same file
            if os.path.abspath(pkl_file) != out_file:
                os.remove(pkl_file)
    #if returning error
    if return_error:
        return error  

def split_tracks_dbscan(feature_cols, 
                        out_pth, 
                        remove, 
                        eps, 
                        min_samples,
                        kwargs,
                        args):
    #split the args
    pkl_files, tracks = args
    #get agent with collected unsplit points
    agent = utils.collect_agent_pkls(pkl_files)
    #split the tracks using kmeans clustering
    agent = agent.split_tracks_dbscan(feature_cols=feature_cols,
                                      tracks=tracks,
                                      eps=eps,
                                      min_samples=min_samples,
                                      **kwargs)
    #now rewrite the pickles
    new_name = f'{agent.meta["Agent ID"]}.tracks'
    out_file = os.path.abspath(os.path.join(out_pth, new_name))
    #add file to meta
    agent.agent_meta['File'] = out_file
    # rewrite pickles
    utils.save_pkl(out_file, agent)
    #if removing unsplit data after splitting
    if remove:
        for pkl_file in pkl_files:
            #incase it rewrote the same file
            if os.path.abspath(pkl_file) != out_file:
                os.remove(pkl_file)

def clip_to_box(files, 
                bbox, 
                col_mapper=mappers.columns,
                out_pth='.', 
                ncores=1, 
                pattern='_clipped'):
    if isinstance(bbox, (Polygon)):
        bbox = bbox.bounds
    elif len(bbox)==4 and isinstance(bbox, (tuple,list,np.ndarray)):
        x0 = bbox[0]
        x1 = bbox[2]
        y0 = bbox[1]
        y1 = bbox[3]
        msg = 'bbox must be shapely Polygon, box or list of (xmin, ymin, xmax, ymax)'
        assert x1>x0 and y1>y0, msg
    else:
        raise Exception('bbox must be shapely Polygon, box or list of (xmin, ymin, xmax, ymax)')
    clip_to_shape(files, 
                  bbox, 
                  out_pth=out_pth, 
                  ncores=ncores, 
                  poly=False, 
                  pattern=pattern,
                  col_mapper=col_mapper)

def clip_to_polygon(files, 
                    poly, 
                    col_mapper=mappers.columns,
                    out_pth='.', 
                    ncores=1, 
                    pattern='_clipped'):
    if isinstance(poly, Polygon):
        poly = np.array(poly.exterior.xy).T
        edges = [(i,i+1) for i in range(len(poly)-1)]
    elif isinstance(poly, np.ndarray):
        edges = [(i,i+1) for i in range(len(poly)-1)]
        pass
    else:
        raise Exception('Polygon must be shapely polygon or 2D numpy array of polygon exterior pts')
    poly = (poly, edges)
    clip_to_shape(files, 
                  poly, 
                  out_pth=out_pth, 
                  ncores=ncores, 
                  poly=True, 
                  pattern=pattern,
                  col_mapper=col_mapper)

def clip_to_shape(files, 
                  extent, 
                  out_pth='.', 
                  ncores=1, 
                  poly=True, 
                  col_mapper=mappers.columns,
                  pattern='_clipped'):   
    if ncores > len(files):
        ncores = len(files)
    else:
        pass
    keep_files = [f for f in files if f.lower().endswith('.csv')]
    missed_files = [f for f in files if f not in keep_files]
    for missed_file in missed_files:
        print(f'Raw file "{missed_file}" is not a csv, skipping...')
    out_files = []
    for raw_file in keep_files:
        out_files.append(os.path.join(out_pth,
                                      os.path.basename(raw_file).replace('.csv',f'{pattern}.csv')))
    clip_func = partial(_clip_to_shape, extent, poly, col_mapper)
    args = list(zip(out_files, files))
    with mp.Pool(ncores) as pool:
        args = list(zip(out_files, files))
        pool.starmap(clip_func, args)

def _clip_to_shape(extent, 
                   poly, 
                   col_mapper,
                   out_file, 
                   raw_file):
    print('Clipping', raw_file)
    cols = pd.read_csv(raw_file, nrows=0)
    #get standard cols and navstats
    new_cols = []
    for col in cols:
        new_cols.append(col_mapper.get(col, col))
    #read the data
    dat = pd.read_csv(raw_file, on_bad_lines='warn')
    if len(dat) == 0:
        return
    else:
        dat.columns = new_cols
        if poly:
            #get the coords
            coords = dat[['X','Y']].values
            #check in
            isin, ison = inpoly2(coords, *extent)
            mask = isin | ison
        else:
            #get the bbox mask
            xmin, ymin, xmax, ymax = extent
            mask1 = (dat['X'] >= xmin) & (dat['X'] <= xmax)
            mask2 = (dat['Y'] >= ymin) & (dat['Y'] <= ymax)
            mask = mask1.values & mask2.values
        dat[mask].to_csv(out_file)

def repair_tracks_spatiotemporal(time_threshold, 
                                 dist_threshold, 
                                 out_pth,
                                 pkl_file):
    drop = []
    refresh = False
    #read the file
    agent = utils.read_pkl(pkl_file)
    #dont stop until the tracks stop joining
    while True:
        #loop over tracks, check for tids to drop
        i = 0 
        tids = list(agent.tracks.keys())
        #list of track ids to join
        drops = []
        #dont stop until until of track
        while True:
            #if end of track
            if i+2 >= len(tids):
                break
            #look at two separated tracks (e.g. 0 and 2)
            else:
                t1 = agent.tracks[tids[i]]
                t2 = agent.tracks[tids[i+2]]
                dt = (t2['Time'].iloc[0] - t1['Time'].iloc[-1]).total_seconds()
                dx = t2['X'].iloc[0] - t1['X'].iloc[-1]          
                dy = t2['Y'].iloc[0] - t1['Y'].iloc[-1]   
                dr = (dx**2+dy**2)**0.5
                #if separated tracks meet joining criteria
                if dt <= time_threshold and dr <= dist_threshold:
                    #drop the middle segment
                    drops.append(tids[i+1])
                    i += 2
                else:
                    i += 1
        #if we need to drop some tracks
        if len(drops) > 0:
            #mark to refresh
            refresh = True
            #remove the tracks we dont want
            for drop in drops:
                agent.tracks.pop(drop)
            #resplit the tracks
            agent = agent.split_tracks_spatiotemporal(time_threshold, 
                                                        dist_threshold)
        #otherwise finish
        else:
            break
    #overwrite the file
    if refresh:
        out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
        agent.agent_meta['File'] = out_file
        utils.save_pkl(out_file, agent)
    
def get_track_gaps(pkl_file):
    #read the file
    agent = utils.read_pkl(pkl_file)
    #init rows list
    rows = []
    #loop over tracks, get the gaps between
    tids = list(agent.tracks.keys())
    if len(tids) > 1:
        for tid1, tid2 in zip(tids[:-1], tids[1:]):
            t1 = agent.tracks[tid1].iloc[-1]['Time']
            t2 = agent.tracks[tid2].iloc[0]['Time']
            dt = (t2-t1).total_seconds()
            dx = agent.tracks[tid1].iloc[-1]['X'] - agent.tracks[tid2].iloc[0]['X']
            dy = agent.tracks[tid1].iloc[-1]['Y'] - agent.tracks[tid2].iloc[0]['Y']
            dr = (dx**2+dy**2)**0.5
            row = {'Track ID_0': f"{agent.meta['Agent ID']}_{tid1}",
                   'Track ID_1': f"{agent.meta['Agent ID']}_{tid2}",
                   'Time Difference': dt,
                   'Distance': dr,
                   'Speed': dr/dt,
                   'File': pkl_file,
                   **agent.meta}
            rows.append(row)
    return rows

def remove_tracks(out_pth,
                  args):
    #separate the args
    pkl_file, tracks = args
    pkl_file = pkl_file[0] #because it comes as a list
    #read the file
    agent = utils.read_pkl(pkl_file)
    #loop over tracks
    keys = list(agent.tracks.keys())
    for tid in keys:
        if f"{agent.meta['Agent ID']}_{tid}" in tracks:
            agent.tracks.pop(tid)
    #if no tracks left, delete it
    if len(agent.tracks) == 0:
        os.remove(pkl_file)
    #otherwise save the updated file
    else:
        out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
        utils.save_pkl(out_file, agent)
        
################################################################################