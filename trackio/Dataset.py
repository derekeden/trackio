################################################################################

from . import (utils, 
               process, 
               geometry, 
               classify, 
               io,
               maps
               )

from .Agent import (gen_track_meta, 
                    gen_agent_meta)

from copy import deepcopy
import re
import pyproj
import geopandas as gp
import glob
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import (box, 
                              LineString, 
                              MultiLineString, 
                              Polygon, 
                              MultiPolygon)
from tqdm import tqdm
import dask.bag as db
import rasterio as rio
from inpoly import inpoly2

################################################################################

agent_database = 'agent.db'
track_database = 'track.db'
dataset_meta = 'dataset.db'

class Dataset:
    
    def __init__(self,
                data_path='./data',
                raw_files=None,
                raw_df=None,
                raw_gdf=None,
                data_files=None,
                meta={}):
        #set attributes
        self.data_path = os.path.abspath(data_path)
        self.meta = self.check_meta(meta)
        #set raw df
        msg = 'raw_df must be None or pd.DataFrame'
        assert isinstance(raw_df, (type(None), pd.DataFrame)), msg
        self.raw_df = raw_df
        #set raw gdf
        msg = 'raw_gdf must be None or gp.GeoDataFrame'
        assert isinstance(raw_gdf, (type(None), gp.GeoDataFrame)), msg
        self.raw_gdf = raw_gdf
        #grab all files if not provided
        data_files = self.check_files(data_files)
        #set status
        self.status = self.init_status(raw_files, 
                                       data_files)
    
    def __repr__(self):
        submeta = self.meta.copy()
        submeta.pop('Static Data')
        submeta.pop('Dynamic Data')
        m = [str(key)+': '+str(val)+'\n' for key,val in submeta.items()]
        type_repr = f'Type:\n    {self.__class__}\n'
        status_repr1 = f'Status:\n'
        status_repr2 = f"    {len(self.status['Unprocessed'])} Unprocessed CSV Files\n"
        status_repr3 = f"    {len(self.status['Processed'])} Processed CSV Files\n"
        status_repr4 = f"    {len(self.status['Unsplit'])} Unsplit Agent Files\n"
        status_repr5 = f"    {len(self.status['Split'])} Split Agent Files\n"
        data_repr1 = f"Static Data Fields:\n    {self.meta['Static Data']}\n"
        data_repr2 = f"Dynamic Data Fields:\n    {self.meta['Dynamic Data']}\n"
        meta_repr = f"Metadata:\n    {'    '.join(m)}"
        path_repr = f'Data Path:\n    {os.path.abspath(self.data_path)}'
        return (type_repr + 
                status_repr1 + 
                status_repr2 + 
                status_repr3 + 
                status_repr4 + 
                status_repr5 + 
                data_repr1 + 
                data_repr2 + 
                meta_repr + 
                path_repr)
        
    ############################################################################
    #ATTRIBUTES
    ############################################################################
    
    #vessel database
    @property
    def agents(self):
        try:
            return utils.read_pkl(f'{self.data_path}/{agent_database}')
        except FileNotFoundError:
            msg = f"{self.data_path}/{agent_database} not found, run self.refresh_meta() first..."
            print(msg)
       
    #track database
    @property
    def tracks(self):
        try:
            return utils.read_pkl(f'{self.data_path}/{track_database}')
        except FileNotFoundError:
            msg = f"{self.data_path}/{track_database} not found, run self.refresh_meta() first..."
            print(msg)
    
    #file mapper
    @property
    def file_mapper(self):
        #get the files and make mapper
        unsplit = self.status['Unsplit']
        split = self.status['Split']
        mapper = {'Agents':{},
                  'Tracks':{}}
        #loop over unsplit (agent) files
        for file in unsplit+split:
            if file.endswith('.points'):
                aid = os.path.basename(file).split('_processor')[0]
            else:
                aid = os.path.basename(file).split('.tracks')[0]
            if aid not in mapper['Agents'].keys():
                mapper['Agents'][aid] = [os.path.abspath(file)]
            else:
                mapper['Agents'][aid].append(os.path.abspath(file))
        #loop over split (track) files
        for file in split:
            aid = os.path.basename(file).split('.tracks')[0]
            mapper['Tracks'][aid] = os.path.abspath(file)
        return mapper
    
    ############################################################################
    #METHODS
    ############################################################################
   
    def check_files(self, data_files):
        if data_files is None:
            data_files = (glob.glob(f'{self.data_path}/*.tracks')+
                         glob.glob(f'{self.data_path}/*.points'))
        else:
            pass
        return list(map(os.path.abspath, data_files))
            
    #init status of raw_files to be processed, pkl_files to be split
    def init_status(self, raw, data):
        #make a dictionary
        status = {'Processed':[],
                  'Unprocessed':[],
                  'Split':[],
                  'Unsplit':[]}
        #add raw files
        if raw is not None:
            status['Unprocessed'].extend(set(raw))
        #add data files
        if data is not None:
            data = set(data)
            #add pkl files, check for pings vs tracks in file name
            for p in data:
                if p.endswith('.tracks'):
                    status['Split'].append(p)
                elif p.endswith('.points'):
                    status['Unsplit'].append(p) 
        return status

    #refresh status from reading the data_path
    def refresh_status(self):
        status = self.init_status(None, self.check_files(None))
        return status    
    
    def get_files_tracks_to_process(self, agents, tracks):
        #get the files to process
        all_files = self.file_mapper
        pkl_files = []
        #based on agent ids
        if agents is not None:
            for agent in agents:
                pkl_files.extend(all_files['Agents'][agent])
                tracks = None
        #based on track ids
        elif tracks is not None:
            for track in tracks:
                pkl_files.append(all_files['Tracks'][track.rsplit('_', 1)[0]])
        #assuming all data
        else:
            pkl_files = self.status['Split'].copy() + self.status['Unsplit'].copy()
            tracks = None
        #group them by agent id
        df = pd.DataFrame({'files':pkl_files, 'track': tracks})
        g = df['files'].tolist()
        g = [os.path.splitext(f)[0] for f in g]
        for i,_g in enumerate(g):
            replace = re.search('_processor[0-9]*', _g)
            if replace:
                g[i] = _g.replace(replace.group(), '')
        df['grouper'] = g
        grouped = df.groupby('grouper')[['files','track']].agg(list)
        grouped['files'] = grouped['files'].apply(np.unique)
        grouped['track'] = grouped['track'].apply(lambda x: np.unique(pd.Series(x).dropna()))
        #return files and track ids corresponding to files
        return grouped['files'].tolist(), grouped['track'].tolist()

    def update_meta(self, out_pth, meta):
        #if written to data_pth, update self.meta
        if out_pth == self.data_path:
            self.meta = meta
        #if written elsewhere, export the meta to there
        else:
            utils.save_pkl(f'{out_pth}/{dataset_meta}', meta)
    
    ############################################################################
    #PREPROCESSING
    ############################################################################

    def group_points(self, 
                     col_mapper={},
                     meta_cols=[],
                     data_cols=['Time','X','Y'],
                     data_mappers={},
                     groupby='MMSI',
                     chunksize=1e6,
                     continued=False,
                     prefix='agent',
                     ncores=1,
                     desc='Grouping points'):
        #create folder if doesn't exist
        #if does, prompt user to delete, or pass continued kwarg
        out_pth = os.path.abspath(self.data_path)
        if continued:
            pass
        else:
            if not os.path.exists(out_pth):
                os.mkdir(out_pth)
            else:
                raise Exception(f'self.data_path already exists, '\
                                 'delete or pass continue=True to resume '\
                                 'processing in this folder...')
        #get the list of raw files
        raw_files = self.status['Unprocessed'].copy()
        #setup partials for function
        partials = (groupby,
                    chunksize,
                    out_pth,
                    col_mapper,
                    meta_cols,
                    data_cols,
                    data_mappers,
                    prefix)
        #if any raw files
        if len(raw_files) > 0:
            #process in parallel
            utils.pool_caller(process.group_points, 
                              partials, 
                              raw_files, 
                              desc, 
                              ncores)
        #check if raw dataframe passed
        if self.raw_df is not None:
            #split into chunks
            chunk_idxs = utils.split_list(list(range(len(self.raw_df))), 
                                          round(len(self.raw_df)/ncores))
            chunks = [self.raw_df.iloc[idx] for idx in chunk_idxs]
            #process in parallel
            utils.pool_caller(process.group_points, 
                              partials, 
                              chunks, 
                              desc,
                              ncores)
            #erase the df
            self.raw_df = None
        #check if raw geodataframe passed
        if self.raw_gdf is not None:
            #split into chunks
            raw_df = utils.gdf_to_df(self.raw_gdf)
            chunk_idxs = utils.split_list(list(range(len(raw_df))), 
                                          round(len(raw_df)/ncores))
            chunks = [raw_df.iloc[idx] for idx in chunk_idxs]
            #process in parallel
            utils.pool_caller(process.group_points, 
                              partials, 
                              chunks, 
                              desc, 
                              ncores)
            #erase the gdf
            self.raw_gdf = None
        #refresh the status
        self.status = self.refresh_status()
        self.status['Unprocessed'] = []
        self.status['Processed'].extend(raw_files)
        return self

    def split_tracks_spatiotemporal(self,
                                    agents=None,
                                    tracks=None,
                                    time=3600 * 12, #seconds, 12hrs
                                    distance=0.5, #same units as DataSet, default = 0.5deg = 55km
                                    ncores=1, 
                                    out_pth=None,
                                    remove=True,
                                    desc='Splitting tracks using spatiotemporal threshold'): 
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process tracks in parallel
        utils.pool_caller(process.split_tracks_spatiotemporal,
                          (time, distance, out_pth, remove, 0), #split method
                          pkl_groups,
                          desc, 
                          ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self
    
    def split_tracks_by_data(self,
                             agents=None,
                             tracks=None,
                             data_col='Status',
                             ncores=1, 
                             out_pth=None,
                             remove=True,
                             desc='Splitting tracks by changes in data column'): 
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process tracks in parallel
        utils.pool_caller(process.split_tracks_by_data,
                          (data_col, out_pth, remove),
                          pkl_groups,
                          desc, 
                          ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self
    
    def split_overlapping_tracks_spatiotemporal(self,
                                    agents=None,
                                    tracks=None,
                                    time=3600 * 12, #seconds, 12hrs
                                    distance=0.5, #same units as DataSet, default = 0.5deg = 55km
                                    ncores=1, 
                                    out_pth=None,
                                    remove=True,
                                    desc='Splitting overlapping tracks using spatiotemporal threshold'): 
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process tracks in parallel
        utils.pool_caller(process.split_tracks_spatiotemporal,
                          (time, distance, out_pth, remove, 1), #split method
                          pkl_groups,
                          desc, 
                          ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self
    
    def split_tracks_kmeans(self,
                            agents=None,
                            tracks=None,
                            n_clusters=range(10),
                            feature_cols=['X','Y'],
                            out_pth=None,
                            ncores=4,
                            return_error=False,
                            remove=True,
                            desc='Using KMeans clustering to split tracks',
                            optimal_method='davies-bouldin',
                            **kwargs):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #make sure k is int or list of ints
        msg = 'n_clusters must be int, list of ints, or range'
        assert (isinstance(n_clusters, int) or 
                (isinstance(n_clusters, list) and all([isinstance(_k, int) for _k in n_clusters])) or
                isinstance(n_clusters, range)), msg
        #turn to list for function
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]
        else:
            n_clusters = sorted(n_clusters)
        #assert more than 2 clusters
        msg = 'n_clusters must be 2 or greater'
        assert min(n_clusters) > 1, msg
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process clustering in parallel
        error = utils.pool_caller(process.split_tracks_kmeans,
                                  (n_clusters, 
                                   feature_cols, 
                                   out_pth, 
                                   return_error, 
                                   remove, 
                                   optimal_method,
                                   kwargs),
                                  pkl_groups,
                                  desc,
                                  ncores)
        #refresh the status
        self.status = self.refresh_status()
        #return
        if return_error:
            error = utils.flatten(error)
            return self, pd.DataFrame(error)
        else:
            return self
    
    def split_tracks_dbscan(self,
                            agents=None,
                            tracks=None,
                            feature_cols=['X','Y'],
                            out_pth=None,
                            ncores=4,
                            remove=True,
                            eps=0.5,
                            min_samples=2,
                            desc='Using DBSCAN clustering to split tracks',
                            **kwargs):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process clustering in parallel
        utils.pool_caller(process.split_tracks_dbscan,
                            (feature_cols, 
                            out_pth, 
                            remove, 
                            eps, 
                            min_samples,
                            kwargs),
                            pkl_groups,
                            desc,
                            ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self
    
    def repair_tracks_spatiotemporal(self, 
                                     agents=None,
                                     time=3600 * 12, #seconds, 12hrs
                                     distance=0.5, #same units as DataSet, default = 0.5deg = 55km
                                     ncores=1, 
                                     out_pth=None,
                                     desc='Repairing tracks using spatiotemporal threshold'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #process repairs in parallel
        utils.pool_caller(process.repair_tracks_spatiotemporal,
                          (time, 
                           distance, 
                           out_pth),
                          pkl_files,
                          desc,
                          ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self
    
    def remove_agents(self,
                      agents=None,
                      desc='Removing agents from database',
                      ncores=1):
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #remove the files in parallel
        utils.pool_caller(os.remove, 
                          (), 
                          pkl_files, 
                          desc, 
                          ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self            
    
    def remove_tracks(self,
                      tracks=None,
                      ncores=1,
                      out_pth=None,
                      desc='Removing tracks from agent files'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(None, tracks)))
        utils.pool_caller(process.remove_tracks,
                          (out_pth,),
                          pkl_groups,
                          desc,
                          ncores)
        #refresh the status
        self.status = self.refresh_status()
        return self
        
    def get_track_splits(self, 
                         agents=None,
                         ncores=1,
                         desc='Getting track split data'):
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #loop through each, process to split tracks
        rows = utils.pool_caller(process.get_track_gaps,
                                 (),
                                 pkl_files,
                                 desc,
                                 ncores)
        rows = utils.flatten(rows)
        df = pd.DataFrame(rows)
        df['Connect'] = False
        return df

    ############################################################################
    #METADATA
    ############################################################################

    def check_meta(self, meta):
        #try to fetch the meta if already exists
        if meta == {}:
            try:
                meta = utils.read_pkl(f'{self.data_path}/{dataset_meta}')
            except FileNotFoundError:
                msg = f'\nNo {dataset_meta} found in {self.data_path}. '\
                       '\nUsing default units/crs. '\
                       '\nEdit self.meta and then run self.refresh_meta to update.\n'
                print(msg)
                meta = utils.std_meta()
        return meta
    
    def refresh_meta(self, 
                     agents_only=False, 
                     ncores=1,
                     desc='Refreshing metadata'):
        #process metadata in parallel
        iterable = self.status['Unsplit'] + self.status['Split']
        #assert there's even data
        assert len(iterable) > 0, f'No point or track files in {self.data_path}'
        meta_metacols_datacols = utils.pool_caller(_refresh_meta,
                                                    (agents_only,),
                                                    iterable,
                                                    desc,
                                                    ncores)
        _meta = [m[0] for m in meta_metacols_datacols]
        _meta_cols = np.unique(utils.flatten([m[1] for 
                                              m in meta_metacols_datacols])).tolist()
        _data_cols = np.unique(utils.flatten([m[2] for 
                                              m in meta_metacols_datacols])).tolist()
        #combine into 1 dict
        meta = {}
        for m in _meta:
            meta |= m
        #get and write agent database
        agents = _meta_to_agents(meta, self.meta['CRS'])
        #convert datetimes
        agents['Start Time'] = pd.to_datetime(agents['Start Time'])
        agents['End Time'] = pd.to_datetime(agents['End Time'])
        #save agent database
        utils.save_pkl(f'{self.data_path}/{agent_database}', agents)
        #if there's track data
        if len(self.status['Split']) > 0:
            #get and write track database
            tracks = _meta_to_tracks(meta, self.meta['CRS'])
            #convert the datetimes
            tracks['Start Time'] = pd.to_datetime(tracks['Start Time'])
            tracks['End Time'] = pd.to_datetime(tracks['End Time'])
            #save track database
            utils.save_pkl(f'{self.data_path}/{track_database}', tracks)
        else:
            #if only unsplit point files
            print(f'No split track files found, skipping {track_database}')
        #save the latest dataset meta
        meta = self.meta
        meta['Static Data'] = _meta_cols
        meta['Dynamic Data'] = _data_cols
        utils.save_pkl(f'{self.data_path}/{dataset_meta}', meta)
        #return
        print(f'New meta/databases saved to {self.data_path}')
        return self

    def make_meta_mapper(self,
                         inp,
                         agents=None,
                         ncores=1,
                         desc='Making meta mappers'):
        #make list
        if isinstance(inp, str):
            inp = [inp]
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #read data from agents in parallel
        _out =  utils.pool_caller(maps.make_meta_mapper,
                                 (inp,),
                                 pkl_files,
                                 desc,
                                 ncores)
        #reduce to unique dictionary
        out = utils.flatten_dict_unique(_out)
        #turn into blank mapper
        maps = [{k:None for k in out[key]} for key in inp]
        #if single mapper just return it
        if len(maps) == 1:
            return maps[0]
        #otherwise nested dict
        else:
            return dict(zip(inp, maps))

    def drop_meta(self, 
                  inp,
                  agents=None,
                  out_pth=None,
                  ncores=1,
                  desc='Dropping meta from agents'):
        #make list
        if isinstance(inp, str):
            inp = [inp]
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #delete data from agents in parallel
        utils.pool_caller(maps.drop_agent_meta,
                        (inp, out_pth),
                        pkl_files,
                        desc,
                        ncores)
        return self
        
    def map_meta(self,
                inp, 
                out,
                mapper,
                agents=None,
                out_pth=None,
                ncores=1,
                drop=False,
                fill=None,
                desc='Mapping metadata to agent'):
        #if only 1 
        if isinstance(inp, str):
            mapper = {inp:mapper}
            msg = 'out should be same type as inp (str)'
            assert isinstance(out, str), msg
            inp = [inp]
            out = [out]
        else:
            msg = f'mapper missing one or all of keys: {inp}'
            assert all([i in mapper.keys() for i in inp]), msg
            msg = 'out is not the same length as inp'
            assert len(inp) == len(out), msg
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #map metadata to agents in parallel
        utils.pool_caller(maps.map_agent_meta,
                          (inp, out, mapper, out_pth, drop, fill),
                          pkl_files,
                          desc,
                          ncores)
        return self
    
    def make_data_mapper(self,
                         inp,
                         agents=None,
                         tracks=None,
                         ncores=1,
                         desc='Making data mappers'):
        #make list
        if isinstance(inp, str):
            inp = [inp]
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #read data from agents in parallel
        _out =  utils.pool_caller(maps.make_data_mapper,
                                 (inp,),
                                 pkl_groups,
                                 desc,
                                 ncores)
        #reduce to unique dictionary
        out = utils.flatten_dict_unique(_out)
        #turn into blank mapper
        maps = [{k:None for k in out[key]} for key in inp]
        #if single mapper just return it
        if len(maps) == 1:
            return maps[0]
        else:
            return dict(zip(inp, maps))
    
    def drop_data(self, 
                  inp,
                  agents=None,
                  out_pth=None,
                  ncores=1,
                  desc='Dropping dynamic data from agents'):
        #make list
        if isinstance(inp, str):
            inp = [inp]
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #delete data from agents in parallel
        utils.pool_caller(maps.drop_agent_data,
                        (inp, out_pth),
                        pkl_files,
                        desc,
                        ncores)
        return self
    
    def map_data(self,
                inp,
                out,
                mapper,
                agents=None,
                tracks=None,
                out_pth=None,
                ncores=1,
                drop=False,
                fill=None,
                desc='Mapping agent dynamic data'):
        #if only 1 
        if isinstance(inp, str):
            mapper = {inp:mapper}
            msg = 'out should be same type as inp (str)'
            assert isinstance(out, str), msg
            inp = [inp]
            out = [out]
        else:
            msg = f'mapper missing one or all of keys: {inp}'
            assert all([i in mapper.keys() for i in inp]), msg
            msg = 'out is not the same length as inp'
            assert len(inp) == len(out), msg
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #map agent data in parallel
        utils.pool_caller(maps.map_agent_data,
                          (inp, out, mapper, out_pth, drop, fill),
                          pkl_groups,
                          desc,
                          ncores)
        return self

    def map_data_to_codes(self,
                          inp,
                          mapper,
                          agents=None,
                          tracks=None,
                          out_pth=None,
                          ncores=1,
                          drop=False,
                          fill=-1,
                          desc='Mapping agent dynamic data to coded boolean arrays'):
        #if only 1 
        if isinstance(inp, str):
            mapper = {inp:mapper}
            inp = [inp]
        else:
            msg = f'mapper missing one or all of keys: {inp}'
            assert all([i in mapper.keys() for i in inp]), msg
        #assert all mapper values are integer and unique
        msg = 'All mapper values must be unique integer codes'
        for m in mapper.values():
            assert all([isinstance(v, int) for v in m.values()]), msg
            assert len(np.unique(list(m.values()))) == len(m.values()), msg
        #check fill value
        assert isinstance(fill, int), 'fill value must be integer'
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #map agent data in parallel
        utils.pool_caller(maps.map_agent_data_to_codes,
                          (inp, mapper, out_pth, drop, fill),
                          pkl_groups,
                          desc,
                          ncores)
        return self

    def set_out_pth(self, out_pth):
        if out_pth is None:
            out_pth = self.data_path
        else:
            if not os.path.isdir(out_pth):
                os.mkdir(out_pth)
        return out_pth

    ############################################################################
    #FETCHERS
    ############################################################################

    #make deepcopy
    def copy(self):
        return deepcopy(self)
    
    #get one agent
    def get_agent(self, agent):
        return utils.collect_agent_pkls(self.file_mapper['Agents'][agent])
    
    #get specific agents
    def get_agents(self, agents, ncores=1):
        agent_files = self.file_mapper['Agents']
        files = [agent_files[a] for a in agents]
        #process in parallel
        with mp.Pool(ncores) as pool:
            out = pool.map(utils.collect_agent_pkls, tqdm(files, 
                                                          total=len(files),
                                                          colour='GREEN'))
        return out
    
    #get one track
    def get_track(self, track):
        aid, tid = track.rsplit('_',1)
        file = self.file_mapper['Tracks'][aid]
        a = utils.read_pkl(file)
        return a.tracks[tid]
            
    #get specific tracks
    def get_tracks(self, tracks, ncores=1):
        aids = list(map(lambda x: x.rsplit('_', 1)[0], tracks))
        files = [self.file_mapper['Tracks'][a] for a in aids]
        #process in parallel
        with mp.Pool(ncores) as pool:
            out = pool.map(utils.read_pkl, tqdm(files, 
                                                total=len(files), 
                                                colour='GREEN'))
        #collect tracks
        out_tracks = []
        for ves in out:
            for tid in ves.tracks.keys():
                if f"{ves.meta['Agent ID']}_{tid}" in tracks:
                    out_tracks.append(ves.tracks[tid])
        return out_tracks

    def to_gdf(self, 
               agents=None,
               tracks=None,
               code=None,
               ncores=1,
               segments=False,
               method='middle',
               desc='Converting tracks to GeoDataFrame'):
        #change desc
        if segments:
            desc = 'Converting track segments to GeoDataFrame'
        #assert method is valid
        if segments:
            msg = 'method must be backward, middle, or forward'
            assert method in ['backward','middle','forward'], msg
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))  
        #process to gdf in parallel
        rows = utils.pool_caller(io.to_gdf,
                                 (code, segments, method),
                                 pkl_groups,
                                 desc,
                                 ncores)
        #flatten and convert to gdf
        rows = utils.flatten(rows)
        #if empty
        if len(rows) == 0:
            gdf = gp.GeoDataFrame()
        else:
            gdf = gp.GeoDataFrame(rows, 
                                  crs=self.meta['CRS'], 
                                  geometry='geometry')
            #make index column
            if segments:
                gdf.index = gdf['Segment ID'].values
            else:
                gdf.index = gdf['Track ID'].values
        return gdf
    
    def to_df(self,
              agents=None,
              tracks=None,
              code=None,
              ncores=1,
              desc='Converting tracks to DataFrame'):
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))  
        #process to gdf in parallel
        dfs = utils.pool_caller(io.to_df,
                                (code,),
                                pkl_groups,
                                desc,
                                ncores)
        #flatten and convert to df
        dfs = utils.flatten(dfs)
        return pd.concat(dfs)
    
    def to_dask_bag(self, 
                    agents=None):
        #get the files to process
        pkl_files, _ = self.get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #create bag
        bag = db.from_sequence(pkl_files)
        return bag.map(utils.read_pkl)

    ############################################################################
    #GEOMETRIC
    ############################################################################
    
    def reproject_crs(self, 
                      crs2,
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      out_pth=None,
                      desc='Reprojecting CRS'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #get crs
        crs1 = CRS(self.meta['CRS'])
        crs2 = CRS(crs2)
        #if already same crs
        if crs1.equals(crs2):
            pass
        #if different crs
        else:
            #make transformer once
            transformer = Transformer.from_crs(crs1, crs2, always_xy=True)
            #reproject in parallel
            utils.pool_caller(geometry.reproject_crs,
                                (transformer, out_pth),
                                pkl_groups,
                                desc, 
                                ncores)
            #reset/write the meta
            meta = self.meta.copy()
            meta['CRS'] = crs2
            meta['X'] = crs2.axis_info[0].unit_name
            meta['Y'] = crs2.axis_info[1].unit_name
            #update meta
            self.update_meta(out_pth, meta)
        return self
         
    def resample_spacing(self, 
                         spacing, 
                         agents=None,
                         tracks=None, 
                         ncores=1, 
                         out_pth=None,
                         desc='Resampling track spacing'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.resample_spacing,
                          (spacing, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
        
    def resample_time(self, 
                      seconds, 
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      out_pth=None,
                      desc='Resampling track timing'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.resample_time,
                          (seconds, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
    
    def resample_time_global(self, 
                             time, 
                             agents=None,
                             tracks=None, 
                             ncores=1, 
                             out_pth=None,
                             desc='Resampling tracks to global time axis'):
        #convert incase
        time = pd.to_datetime(time)
        #assert entire dataset is spanned
        adb = self.agents
        start = adb['Start Time'].min()
        end = adb['End Time'].max()
        msg = f'Global time must span entire dataset from {start} to {end} inclusive'
        assert start >= time[0] and end <= time[-1], msg
        #convert times to seconds
        _time = time.astype(np.int64)/1e9 #seconds
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.resample_time_global,
                          (_time, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self    

    def compute_coursing(self, 
                         agents=None,
                         tracks=None, 
                         ncores=1, 
                         out_pth=None,
                         method='middle',
                         desc='Computing coursing'):
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_coursing,
                          (method, self.meta['CRS'], out_pth),
                          pkl_groups,
                          desc,
                          ncores)       
        #update meta
        meta = self.meta.copy()
        meta['Coursing'] = 'degrees'
        self.update_meta(out_pth, meta)
        return self
    
    def compute_turning_rate(self, 
                             agents=None,
                             tracks=None, 
                             ncores=1, 
                             out_pth=None,
                             method='middle',
                             desc='Computing turning rate'):
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_turning_rate,
                          (method, out_pth),
                          pkl_groups,
                          desc,
                          ncores)       
        #update meta
        meta = self.meta.copy()
        meta['Turning Rate'] = 'degrees/sec'
        self.update_meta(out_pth, meta)
        return self

    def compute_speed(self, 
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      out_pth=None,
                      method='middle',
                      desc='Computing speed'):
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_speed,
                          (method, out_pth),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Speed'] = f"{meta['X']}/second"
        self.update_meta(out_pth, meta)
        return self

    def compute_acceleration(self, 
                             agents=None,
                             tracks=None, 
                             ncores=1, 
                             out_pth=None,
                             desc='Computing acceleration'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_acceleration,
                          (out_pth,),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Acceleration'] = f"{meta['Speed']}/second"
        self.update_meta(out_pth, meta)
        return self

    def smooth_corners(self, 
                       refinements=2,
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       out_pth=None,
                       desc='Smoothing sharp corners'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.smooth_corners,
                          (refinements, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
            
    def decimate_tracks(self, 
                        epsilon=1,
                        agents=None,
                        tracks=None, 
                        ncores=1, 
                        out_pth=None,
                        desc='Decimating tracks'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.decimate_tracks,
                          (epsilon, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
       
    def characteristic_tracks(self,
                              stop_threshold=0.15,
                              turn_threshold=22.5,
                              min_distance=500,
                              max_distance=20000,
                              min_stop_duration=1800,
                              agents=None,
                              tracks=None, 
                              ncores=1, 
                              out_pth=None,
                              inplace=False,
                              desc='Extracting characteristic tracks'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.characteristic_tracks,
                          (stop_threshold,
                           turn_threshold,
                           min_distance,
                           max_distance,
                           min_stop_duration,
                           out_pth,
                           inplace),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
    
    def simplify_stops(self, 
                       stop_threshold=0.15, 
                       min_stop_duration=1800,
                       max_drift_distance=1000,
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       out_pth=None,
                       desc='Simplifying stops along tracks'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.simplify_stops,
                          (stop_threshold,
                           min_stop_duration,
                           max_drift_distance,
                           out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
    
    def imprint_geometry(self, 
                         shape,
                         agents=None,
                         tracks=None, 
                         ncores=1, 
                         out_pth=None,
                         desc='Imprinting geometry into tracks'):
        #check input is valid
        msg = 'geometry must be a shapely LineString, MultiLineString, Polygon, or MultiPolygon'
        assert isinstance(shape, (LineString, 
                                  MultiLineString, 
                                  Polygon, 
                                  MultiPolygon)), msg
        polylines = utils.prepare_polylines(shape)
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.imprint_geometry,
                          (polylines, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self

    def interpolate_raster(self,
                           ras_file,
                           name,
                           agents=None,
                           tracks=None,
                           ncores=1,
                           out_pth=None,
                           method='linear',
                           desc='Interpolating raster to tracks'):
        #read raster
        ras = rio.open(ras_file)
        #assert same crs
        msg = 'raster file must have same CRS as Dataset'
        assert pyproj.CRS(self.meta['CRS']).equals(ras.crs), msg
        x,_ = ras.xy([0]*ras.shape[0],range(ras.shape[0]))
        _,y = ras.xy(range(ras.shape[1]), [0]*ras.shape[1])
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((x,y), 
                                         ras.read(1),
                                         bounds_error=False,
                                         fill_value=np.nan,
                                         method=method)
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.interpolate_raster,
                          (interp, name, out_pth),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self

    def encounters(self, 
                   dataset=None, 
                   distance=0.1, 
                   time=3600, 
                   ncores=1, 
                   tracks0=None, 
                   tracks1=None,
                   data_cols=['Speed', 'Acceleration', 'Turning Rate', 'Coursing'],
                   meta_cols=['Name'],
                   filter_min=False,
                   desc='Calculating spatiotemporal encounters'):
        #if no input, self encounters
        if dataset is None:
            dataset = self.copy()
        assert self.meta['CRS'] == dataset.meta['CRS'], 'CRS does not match between Datasets'
        #get the two track databases
        db1 = self.tracks
        db2 = dataset.tracks
        #reduce
        keepcols = ['Start Time','End Time', 'geometry', 'File']
        db1 = db1[keepcols]
        db2 = db2[keepcols]
        if tracks0 is not None:
            db1 = db1.loc[tracks0]
        if tracks1 is not None:
            db2 = db2.loc[tracks1]        
        #process in parallel
        rows = utils.pool_caller(geometry.encounters,
                                 (db1, 
                                  db2, 
                                  distance, 
                                  time, 
                                  data_cols,
                                  meta_cols,
                                  filter_min),
                                 list(range(len(db1))),
                                 desc,
                                 ncores)
        #convert to geodataframe and return
        rows = utils.flatten(rows)
        if len(rows)>0:
            out = gp.GeoDataFrame(rows, geometry='geometry', crs=self.meta['CRS'])
        else:
            out = gp.GeoDataFrame()
        return out  
         
    def intersections(self, 
                      dataset=None, 
                      time=3600, 
                      ncores=1, 
                      tracks0=None, 
                      tracks1=None,                   
                      data_cols=['Speed', 'Acceleration', 'Turning Rate', 'Coursing'],
                      meta_cols=['Name'],
                      desc='Calculating intersections'):
        #if no input, self intersection
        if dataset is None:
            dataset = self.copy()
        assert self.meta['CRS'] == dataset.meta['CRS'], 'CRS does not match between DataSets'
        #get the two track databases
        db1 = self.tracks
        db2 = dataset.tracks
        #reduce
        keepcols = ['Start Time','End Time', 'geometry', 'File']
        db1 = db1[keepcols]
        db2 = db2[keepcols]
        if tracks0 is not None:
            db1 = db1.loc[tracks0]
        if tracks1 is not None:
            db2 = db2.loc[tracks1]        
        #process in parallel
        rows = utils.pool_caller(geometry.intersections,
                                 (db1, db2, time, data_cols, meta_cols),
                                 list(range(len(db1))),
                                 desc,
                                 ncores)
        #convert to geodataframe and return
        rows = utils.flatten(rows)
        if len(rows)>0:
            out = gp.GeoDataFrame(rows, geometry='geometry', crs=self.meta['CRS'])
        else:
            out = gp.GeoDataFrame()
        return out   
    
    def proximity_to_object(self, 
                            shapely_object, 
                            agents=None,
                            tracks=None, 
                            data_cols=['Speed', 'Acceleration', 'Turning Rate', 'Coursing'],
                            meta_cols=['Name'],
                            ncores=1, 
                            desc='Calculating proximities to object'):
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #prepare geometry
        shapes = utils.prepare_polylines(shapely_object)
        #get proximities in parallel
        rows = utils.pool_caller(geometry.proximity_to_object,
                                 (shapes, data_cols, meta_cols),
                                 pkl_groups,
                                 desc,
                                 ncores)
        #flatten and return
        rows = utils.flatten(rows)
        out = pd.DataFrame(rows)
        return out
    
    def proximities(self,
                    dataset=None,
                    tracks0=None, 
                    tracks1=None,
                    data_cols=['Speed', 'Acceleration', 'Turning Rate', 'Coursing'],
                    meta_cols=['Name'],
                    ncores=1, 
                    bins=None, 
                    relative=False,
                    desc='Calculating agent proximities'):
        print('Proximity analysis assumes that self.resample_time_global has already been run, '\
            'if not the results will be invalid or the function may fail')
        #if no input, self encounters
        if dataset is None:
            dataset = self.copy()
        assert self.meta['CRS'] == dataset.meta['CRS'], 'CRS does not match between Datasets'
        #get the two track databases
        db1 = self.tracks
        db2 = dataset.tracks
        #reduce
        keepcols = ['Start Time','End Time', 'geometry', 'File']
        db1 = db1[keepcols]
        db2 = db2[keepcols]
        if tracks0 is not None:
            db1 = db1.loc[tracks0]
        if tracks1 is not None:
            db2 = db2.loc[tracks1]        
        #process proximities in parallel
        rows = utils.pool_caller(geometry.proximities,
                                 (db1, 
                                  db2, 
                                  bins, 
                                  relative, 
                                  data_cols, 
                                  meta_cols),
                                 list(range(len(db1))),
                                 desc,
                                 ncores)
        #convert to dataframe
        if relative:
            out = pd.concat([pd.DataFrame(r) for r in rows]).reset_index(drop=True)
            out = out.groupby('Time').agg(max).reset_index()
        else:
            rows = utils.flatten(rows)
            out = pd.DataFrame(rows)
        return out
    
    def lateral_distribution(self,
                             start,
                             end,
                             split=True,
                             spacing=100,
                             n_slices=0,
                             agents=None,
                             tracks=None,
                             ncores=1,
                             polygon=None,
                             density=True,
                             bins=100,
                             meta_cols=[],
                             data_cols=[],
                             desc='Calculating lateral distributions at slices'):
        #length/slope of arc
        dist = (np.diff(np.array([start, end]), axis=0)**2).sum()**0.5
        dy, dx = ((end[1] - start[1])/dist , (end[0] - start[0])/dist)
        #longitudinal slice locations
        long, xy = utils.longitudinal_slices(start, 
                                             end, 
                                             n_slices, 
                                             spacing, 
                                             dist)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        dfs = utils.pool_caller(geometry.lateral_distribution,
                                (long, xy, dy, dx, meta_cols, data_cols),
                                pkl_groups,
                                desc,
                                ncores)
        #flatten and turn into dataframe
        df = pd.concat([_df for _df in dfs if len(_df)>0]).reset_index(drop=True)
        #clip to polygon
        if polygon is not None:
            polygon, edges = utils.format_polygon(polygon)
            mask = np.logical_or(*inpoly2(df[['TrackX', 'TrackY']].values, 
                                          polygon, 
                                          edges))
            df = df.loc[mask]
        #groupby
        if split:
            grouper = ['Longitudinal Distance', 'Direction']
        else:
            grouper = ['Longitudinal Distance']
        grouped = df.groupby(grouper)[df.columns].agg(list)
        #drop and format couple columns
        grouped.drop(columns=['Longitudinal Distance',
                              'Direction'], inplace=True)
        grouped['SliceX'] = grouped['SliceX'].apply(lambda x: x[0])
        grouped['SliceY'] = grouped['SliceY'].apply(lambda x: x[0])
        #get global min/max
        global_min = grouped['Lateral Distance'].explode().min()
        global_max = grouped['Lateral Distance'].explode().max()
        #create global bin edges and add to df
        edges = utils.range_to_edges(global_min, global_max, bins)
        grouped['Bins'] = [edges]*len(grouped)
        #add frequency
        grouped['Frequency'] = grouped.apply(lambda row: np.histogram(row['Lateral Distance'],
                                                                      bins=row['Bins'],
                                                                      density=False)[0], 
                                             axis=1)
        #if density, divide by total number of entries
        if density:
            grouped['Frequency'] /= grouped['Lateral Distance'].apply(len)
        #return
        return grouped
    
    def time_in_polygon(self, 
                        polygon,
                        agents=None,
                        tracks=None,
                        meta_cols=[],
                        data_cols=[],
                        desc='Computing time spent in polygon',
                        ncores=1):
        #get polygon
        polygon, edges = utils.format_polygon(polygon)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #compute in parallel
        rows = utils.pool_caller(geometry.time_in_polygon,
                                (polygon, edges, meta_cols, data_cols),
                                pkl_groups,
                                desc,
                                ncores)
        #flatten and turn into dataframe
        rows = utils.flatten(rows)
        df = pd.DataFrame(rows)
        return df
    
    def generate_flow_map(self,
                          polygons,
                          characteristic_col=None,
                          flow_col='Graph Node',
                          agents=None,
                          tracks=None,
                          ncores=1,
                          desc='Generating flow map from polygons'):
        #make sure it is proper format
        msg = 'Polygons must be gp.GeoDataFrame with Code, X, Y columns'
        assert isinstance(polygons, gp.GeoDataFrame), msg
        assert all([c in polygons.columns for c in ['Code','X','Y']]), msg
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #compute in parallel
        rows = utils.pool_caller(geometry.generate_flow_map,
                                (characteristic_col, 
                                 flow_col),
                                pkl_groups,
                                desc,
                                ncores)
        #flatten and convert to dataframe
        rows = utils.flatten(rows)
        df = pd.DataFrame(rows)
        df = df.groupby(['Start', 'End']).agg(list)
        df['Volume'] = df['Track ID'].apply(len)
        #add the linestrings
        keys = polygons['Code'].values
        vals = polygons[['X','Y']].values
        points = dict(zip(keys, vals))
        geo = []
        for idx in df.index:
            line = LineString([points[idx[0]], points[idx[1]]])
            geo.append(line)
        #turn to gdf
        gdf = gp.GeoDataFrame(df, geometry=geo, crs=polygons.crs)
        #add some properties
        gdf['Length'] = gdf.geometry.length
        dx = gdf.geometry.apply(lambda x: x.coords[-1][0] - x.coords[0][0])
        dy = gdf.geometry.apply(lambda x: x.coords[-1][1] - x.coords[0][1])
        gdf['Direction'] = np.degrees(np.arctan2(dx, dy)) % 360
        return gdf.reset_index()

    def reduce_to_flow_map(self,
                          polygons,
                          characteristic_col='Characteristic',
                          flow_col='Graph Node',
                          agents=None,
                          tracks=None,
                          ncores=1,
                          out_pth=None,
                          desc='Generating flow map from polygons'):
        #set out path
        out_pth = self.set_out_pth(out_pth)
        #make sure it is proper format
        msg = 'Polygons must be gp.GeoDataFrame with Code, X, Y columns'
        assert isinstance(polygons, gp.GeoDataFrame), msg
        assert all([c in polygons.columns for c in ['Code','X','Y']]), msg
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #make point lookup
        keys = polygons['Code'].values
        vals = polygons[['X','Y']].values
        points = dict(zip(keys, vals))
        #compute in parallel
        utils.pool_caller(geometry.reduce_to_flow_map,
                        (characteristic_col, 
                        flow_col,
                        out_pth,
                        points),
                        pkl_groups,
                        desc,
                        ncores)
        #update meta
        self.update_meta(out_pth, self.meta)
        return self
             
    # def flow_map(self):
    #     #implement that paper 
    #     pass
    
    # def reduce_to_flow_map(self):
    #     #implement that paper 
    #     pass

    ############################################################################
    #CLUSTERING
    ############################################################################

    # def MiniBatchKMeans(self, ncores)

    ############################################################################
    #CLASSIFYING
    ############################################################################
                
    def classify_in_polygon(self,
                            polygon,
                            agents=None,
                            tracks=None, 
                            ncores=1, 
                            code=16,
                            desc='Classifying tracks inside polygon'):
        #get polygon
        polygon, edges = utils.format_polygon(polygon)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(classify.classify_in_polygon,
                          (polygon, edges, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = 'Inside Polygon'
        self.meta[f'Code{code}'] = meta
        return self
    
    def classify_in_polygons(self,
                            polygons,
                            agents=None,
                            tracks=None, 
                            ncores=1,
                            to_codes=True,
                            name='Polygon',
                            desc='Classifying tracks inside polygons'):
        #copy
        polygons = polygons.copy()
        #get polygons
        polygons['polys'] = polygons.geometry.apply(utils.format_polygon)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(classify.classify_in_polygons,
                          (polygons, to_codes, name),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta='Inside Polygon'
        for code in polygons['Code'].values:
            self.meta[f'Code{code}'] = meta
        return self
    
    def classify_speed(self, 
                       speed, 
                       higher=True, 
                       lower=False, 
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       code=17,
                       desc='Classifying tracks by speed threshold'):
        #check for bounds
        if lower:
            higher = False
        elif higher:
            lower = False
        else:
            raise Exception('Must pass higher=True or lower=True, but not both')
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(classify.classify_speed,
                          (speed, higher, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = f'Speed {">=" if higher else "<="} {speed}'
        self.meta[f'Code{code}'] = meta
        return self
    
    def classify_turns(self, 
                       rate, 
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       code=18,
                       desc='Classifying turning tracks'):
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(classify.classify_turns,
                          (rate, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = f"Turning rate >= {rate} deg/sec"
        self.meta[f'Code{code}'] = meta
        return self 
    
    def classify_speed_in_polygon(self, 
                                  speed, 
                                  polygon,
                                  higher=True, 
                                  lower=False, 
                                  agents=None,
                                  tracks=None, 
                                  ncores=1, 
                                  code=19,
                                  desc='Classifying tracks by speed threshold in polygon'):
        #get polygon
        polygon, edges = utils.format_polygon(polygon)
        #check for bounds
        if lower:
            higher = False
        elif higher:
            lower = False
        else:
            raise Exception('Must pass higher=True or lower=True, but not both')
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(classify.classify_speed_in_polygon,
                          (speed, polygon, edges, higher, code),
                          pkl_groups,
                          desc,
                          ncores) 
        #update meta
        meta = f'Speed {">=" if higher else "<="} {speed} inside Polygon'
        self.meta[f'Code{code}'] = meta
        return self   
    
    def classify_trip(self, 
                      poly1, 
                      poly2,  
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      code=20,
                      desc='Classifying tracks by trip'):
        #get polygons
        poly1, edges1 = utils.format_polygon(poly1)
        poly2, edges2 = utils.format_polygon(poly2)
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(classify.classify_trip,
                          (poly1, edges1, poly2, edges2, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = 'Trip between 2 polygons'
        self.meta[f'Code{code}'] = meta
        return self 
    
    def classify_touching(self, 
                          geom, 
                          agents=None,
                          tracks=None, 
                          ncores=1, 
                          code=21,
                          desc='Classifying tracks touching object'):
        msg = 'geometry must be shapely LineString, MultiLineString, Polygon, or MultiPolygon'
        assert isinstance(geom, (LineString, 
                                 MultiLineString, 
                                 Polygon, 
                                 MultiPolygon)), msg
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(classify.classify_touching,
                          (geom, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = 'Touching object'
        self.meta[f'Code{code}'] = meta
        return self 

    def classify_stops(self, 
                       stop_threshold=0.15, 
                       min_stop_duration=1800, 
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       code=22,
                       desc='Classifying stopped tracks'):
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(classify.classify_stops,
                          (stop_threshold, min_stop_duration, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = f'Stopped: Speed <= {stop_threshold} and Duration >= {min_stop_duration}'
        self.meta[f'Code{code}'] = meta
        return self 

    def classify_custom(self, 
                        values, 
                        ncores=1, 
                        code=24,
                        desc='Classifying tracks with custom code',
                        meta='Custom classifier'):
        #make sure it contains the column and is boolean
        msg = 'values must be a pd.DataFrame with Track IDs as index, and a "Value" column consisting of booleans'
        assert 'Value' in values.columns and values['Value'].dtype == bool, msg
        #get the files to process
        pkl_groups = list(zip(*self.get_files_tracks_to_process(None, values.index)))
        #process in parallel
        utils.pool_caller(classify.classify_custom,
                          (values, code),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self.meta[f'Code{code}'] = meta
        return self
       
################################################################################

def _meta_to_agents(meta, crs):
    #make dataframe from meta dict
    df = pd.DataFrame(meta).T.set_index('Agent ID')
    #drop track meta dicts
    df.drop(columns=["Tracks"], inplace=True)
    #make bounding box geometry
    _geometry = df[['Xmin',
                   'Ymin',
                   'Xmax',
                   'Ymax']].apply(lambda b: box(b['Xmin'], 
                                               b['Ymin'], 
                                               b['Xmax'], 
                                               b['Ymax']), axis=1)
    #convert to geodataframe
    df = gp.GeoDataFrame(df, geometry=_geometry, crs=crs)
    return df

def _meta_to_tracks(meta, crs):
    #initialize dataframe rows/indices
    rows = []
    # ids = []
    #loop over meta dict, create entries for each track
    for vid, vmeta in meta.items():
        #get the track ids
        # _ids = [f"{vid}_{tid}" for tid in vmeta["Tracks"].keys()]
        #get the combined track meta rows
        _vmeta = {k:v for k,v in vmeta.items() if k != 'Tracks'}
        tmeta = [
            {**_vmeta, **vmeta["Tracks"][tid]}
            for tid in vmeta["Tracks"].keys()
        ]
        # ids.extend(_ids)
        rows.extend(tmeta)
    #create dataframe from rows and indices
    df = pd.DataFrame(rows)
    # df.index = ids
    df.set_index('Track ID', inplace=True)
    #create bounding box geometry
    _geometry = df[['Xmin',
                   'Ymin',
                   'Xmax',
                   'Ymax']].apply(lambda b: box(b['Xmin'], 
                                               b['Ymin'], 
                                               b['Xmax'], 
                                               b['Ymax']), axis=1)
    #drop pointless column
    df.drop(columns=["ntracks"], inplace=True)
    #convert to geodataframe
    df = gp.GeoDataFrame(df, geometry=_geometry, crs=crs)
    return df

def _refresh_meta(agents_only, file):
    #make a new meta dict and track_meta dict
    meta = {}
    track_meta = {}
    #read the vessel
    v = utils.read_pkl(file)
    #loop over tracks, populate track_meta
    tids = v.tracks.keys()
    for tid in tids:
        track_meta[tid] = gen_track_meta(v.tracks[tid])
        track_meta[tid]['Track ID'] = f"{v.meta['Agent ID']}_{tid}"
    #reset vessel.track_meta
    v.track_meta = track_meta
    #reset vessel.vessel_meta
    v.agent_meta = gen_agent_meta(v)
    v.agent_meta['File'] = file
    #update output meta dict
    meta[v.agent_meta['Agent ID']] = v.agent_meta
    meta[v.agent_meta['Agent ID']]['Tracks'] = track_meta
    #if tracks not split, pings exist in pickle chunks - don't overwrite
    if not agents_only:
        utils.save_pkl(file, v)
    meta_cols = list(v.meta.keys())
    data_cols = v.data.columns.tolist()
    return meta, meta_cols, data_cols

################################################################################