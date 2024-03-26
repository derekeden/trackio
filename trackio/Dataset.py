################################################################################

from . import (utils, 
               process, 
               geometry, 
               classify, 
               io,
               maps)

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
import rasterio as rio
import dask.bag as db
from inpoly import inpoly2
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

GREEN = "\033[92m"
ENDC = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[93m"

################################################################################

#hardcoded file names
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
        """
        Initializes a Dataset object.
        
        Args:
            data_path (str): Defaults to './data'.
            raw_files (list, optional): List of raw data files. Defaults to None.
            raw_df (pd.DataFrame, optional): Raw data in dataframe format. Defaults to None.
            raw_gdf (gp.GeoDataFrame, optional): Raw data in geodataframe format. Defaults to None.
            data_files (list, optional): List of processed data files (*.points or *.tracks). Defaults to None.
            meta (dict, optional): Dataset metadata dictionary. Defaults to {}.
        Returns:
            Dataset: Initialized Dataset object.
        """
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
        #grab all processed files if None provided
        data_files = self._check_files(data_files)
        #set file status
        self.status = self._init_status(raw_files, 
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
    
    #agent database
    @property
    def agents(self):
        """
        Agent database. This is the GeoDataFrame stored in self.data_path/agent.db.
        Use this to efficiently query agents.
        """
        try:
            return utils.read_pkl(f'{self.data_path}/{agent_database}')
        except FileNotFoundError:
            msg = RED + f"{self.data_path}/{agent_database} not found, run self.refresh_meta() first" + ENDC
            print(msg)
       
    #track database
    @property
    def tracks(self):
        """
        Track database. This is the GeoDataFrame stored in self.data_path/track.db.
        Use this to efficiently query tracks.
        """
        try:
            return utils.read_pkl(f'{self.data_path}/{track_database}')
        except FileNotFoundError:
            msg = RED + f"{self.data_path}/{track_database} not found, run self.refresh_meta() first..." + ENDC
            print(msg)
    
    #file mapper for each agent/track
    @property
    def _file_mapper(self):
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
   
    def _check_files(self, data_files):
        #if None is passed
        if data_files is None:
            data_files = (glob.glob(f'{self.data_path}/*.tracks')+
                         glob.glob(f'{self.data_path}/*.points'))
        #otherwise use the list of files
        else:
            pass
        #return abspaths
        return list(map(os.path.abspath, data_files))

    def _init_status(self, raw, data):
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
            #add pkl files, check for points vs tracks in file name
            for p in data:
                if p.endswith('.tracks'):
                    status['Split'].append(p)
                elif p.endswith('.points'):
                    status['Unsplit'].append(p) 
        return status

    def _refresh_status(self):
        #refresh file status after processing raw files
        status = self._init_status(None, self._check_files(None))
        return status    
    
    def _get_files_tracks_to_process(self, agents, tracks):
        #get the files to process
        all_files = self._file_mapper
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

    def _set_out_path(self, out_path):
        if out_path is None:
            out_path = self.data_path
        else:
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
        return out_path
    
    def _update_meta(self, out_path, meta):
        #if written to data_pth, update self.meta
        if out_path == self.data_path:
            self.meta = meta
            out = self
        #if written elsewhere, export the meta to there
        else:
            #export meta
            utils.save_pkl(f'{out_path}/{dataset_meta}', meta)
            #replace the output
            out = Dataset(data_path=out_path)
        return out

    ############################################################################
    #PREPROCESSING
    ############################################################################

    def group_points(self, 
                     col_mapper={},
                     meta_cols=[],
                     data_cols=['Time','X','Y'],
                     data_mappers={},
                     groupby='Unique_ID',
                     chunksize=1e6,
                     continued=False,
                     prefix='agent',
                     ncores=1,
                     desc='Grouping points'):
        """
        Groups all points in Dataset based on groupby column(s) to isolate
        the points belonging to each unique agent in the raw data. Grouped
        data is written to *.points files in the Dataset.data_path location.

        Note, ['Time', 'X', 'Y'] is the bare minimum required for data_cols.

        Also, the continued kwarg is used to resume (crashed or previous) analyses
        in the Dataset.data_path folder. By default, this is set to False to avoid
        data corruption or overwriting.

        Args:
            col_mapper (dict, optional): Mapping for column names. Defaults to {}.
            meta_cols (list, optional): List of metadata columns to maintain. Defaults to [].
            data_cols (list): List of data columns to maintain. Defaults to ['Time', 'X', 'Y'].
            groupby (str, list): Column name(s) to group points by. Defaults to 'Unique_ID'.
            data_mappers (dict): Data mappers for raw data transformation during grouping. Defaults to {}.
            chunksize (float): Size of chunks for raw data reading/processing. Defaults to 1e6 rows/entries.
            continued (bool): Flag to indicate continuation of previous processing in Dataset.data_path. Defaults to False.
            prefix (str): Prefix for output files. Defaults to 'agent'.
            ncores (int): Number of cores to use for processing. Defaults to 1.
            desc (str): Description of the operation. Defaults to 'Grouping points'.

        Returns:
            self: Returns the original Dataset instance.
        """
        #create folder if doesn't exist
        #if does, prompt user to delete, or pass continued kwarg
        #this prevents accidentally corrupting/overwriting data
        out_path = os.path.abspath(self.data_path)
        if continued:
            pass
        else:
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            else:
                raise Exception(f'self.data_path already exists, '\
                                 'delete or pass continue=True to resume '\
                                 'processing in this folder...')
        #get the list of raw files
        raw_files = self.status['Unprocessed'].copy()
        #setup partials for function
        partials = (groupby,
                    chunksize,
                    out_path,
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
        self.status = self._refresh_status()
        self.status['Unprocessed'] = []
        self.status['Processed'].extend(raw_files)
        return self

    def split_tracks_spatiotemporal(self,
                                    agents=None,
                                    tracks=None,
                                    time=3600 * 12, #seconds, 12hrs
                                    distance=0.5, #same units as DataSet, default = 0.5deg = 55km
                                    ncores=1, 
                                    out_path=None,
                                    remove=True,
                                    desc='Splitting tracks using spatiotemporal threshold'): 
        """
        Perform standard spatiotemporal split on points to generate tracks. This algorithm
        simply looks at gaps between adjacent points, if the gap in time OR distance is exceeded
        between points, this is considered a split to a new track.

        This can be used to split *.points files for the first time, or resplit tracks another way
        after they've already been split.

        If you pass remove=True, it will delete *.points files as they are split and saved into *.tracks files.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (optional): Specific list of agent ids to be processed.
            tracks (optional): Specific list track ids to be processed.
            time (int, optional): Time threshold in seconds for splitting. Defaults to 3600 * 12 = 12 hours.
            distance (int, float, optional): Distance threshold in the same units as DataSet (default is 0.5, approx. 55km if data is geographic). Defaults to 0.5.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            out_path (str, optional): Output path for the split tracks. Defaults to None (uses self.data_path).
            remove (bool, optional): Whether to remove the original point files after splitting. Defaults to True.
            desc (str, optional): Description of the operation. Defaults to 'Splitting tracks using spatiotemporal threshold'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #process tracks in parallel
        utils.pool_caller(process.split_tracks_spatiotemporal,
                          (time, distance, out_path, remove, 0), #split method
                          pkl_groups,
                          desc, 
                          ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def split_tracks_by_data(self,
                             agents=None,
                             tracks=None,
                             data_col='Status',
                             ncores=1, 
                             out_path=None,
                             remove=True,
                             desc='Splitting tracks by changes in data column'): 
        """
        Splits tracks based on changes in a specified data column. This method is useful for segmenting
        tracks when a particular attribute changes, for example, when an agent's code column changes. This
        function works by splitting tracks when the value in the specified data column changes, e.g. from 
        True to False, from 0 to 1, etc.

        If you pass remove=True, it will delete *.points files as they are split and saved into *.tracks files.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (optional): Specific list of agent ids to be processed. If None, all agents are processed.
            tracks (optional): Specific list of track ids to be processed. If None, all tracks are processed.
            data_col (str, optional): The name of the data column to use as the criterion for splitting tracks. Defaults to 'Status'.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            out_path (str, optional): Output path for the split tracks. If None, uses the original dataset path. Defaults to None.
            remove (bool, optional): Whether to remove the original .points files after splitting and saving the .tracks files. Defaults to True.
            desc (str, optional): Description of the operation for logging or user information. Defaults to 'Splitting tracks by changes in data column'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #process tracks in parallel
        utils.pool_caller(process.split_tracks_by_data,
                          (data_col, out_path, remove),
                          pkl_groups,
                          desc, 
                          ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def split_overlapping_tracks_spatiotemporal(self,
                                    agents=None,
                                    tracks=None,
                                    time=3600 * 12, #seconds, 12hrs
                                    distance=0.5, #same units as DataSet, default = 0.5deg = 55km
                                    ncores=1, 
                                    out_path=None,
                                    remove=True,
                                    desc='Splitting overlapping tracks using spatiotemporal threshold'): 
        """
        Perform standard spatiotemporal split on points to generate tracks. This algorithm
        differs from split_tracks_spatiotemporal but uses the same inputs. This can sometimes
        be used to perform a spatiotemporal split on data where there are duplicate Unique IDs
        present in the data, making an agent look like it's in two places at once.

        This algorithm starts off with 1 track containing the first point. It then loops over the remaining
        points. If the next point falls within the spatiotemporal thresholds of the previous point, it will
        be append to that track. Otherwise, a new track will be created starting with that point. On the
        next point, all existing tracks will be checked before creating a new one. This process
        continues until there are no more points left to append.

        This can be used to split *.points files for the first time, or resplit tracks another way
        after they've already been split.

        If you pass remove=True, it will delete *.points files as they are split and saved into *.tracks files.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (optional): Specific list of agent ids to be processed.
            tracks (optional): Specific list track ids to be processed.
            time (int, optional): Time threshold in seconds for splitting. Defaults to 3600 * 12 = 12 hours.
            distance (int, float, optional): Distance threshold in the same units as DataSet (default is 0.5, approx. 55km if data is geographic). Defaults to 0.5.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            out_path (str, optional): Output path for the split tracks. Defaults to None (uses self.data_path).
            remove (bool, optional): Whether to remove the original point files after splitting. Defaults to True.
            desc (str, optional): Description of the operation. Defaults to 'Splitting tracks using spatiotemporal threshold'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #process tracks in parallel
        utils.pool_caller(process.split_tracks_spatiotemporal,
                          (time, distance, out_path, remove, 1), #split method
                          pkl_groups,
                          desc, 
                          ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def split_tracks_kmeans(self,
                            agents=None,
                            tracks=None,
                            n_clusters=range(10),
                            feature_cols=['X','Y'],
                            out_path=None,
                            ncores=1,
                            return_error=False,
                            remove=True,
                            desc='Using KMeans clustering to split tracks',
                            optimal_method='davies-bouldin',
                            **kwargs):
        """
        Splits tracks using KMeans clustering based on specified features. This method attempts
        to identify natural groupings of data points within tracks based on the specified features
        and separates tracks accordingly. You can optionally output the error metrics using
        return_error=True. This includes the inertia, Davies-Bouldin and Silhouette scores.

        If you pass a list/range of n_clusters to test, you can also specify the optimal_method kwarg
        to choose the optimal number of clusters. Options are ['davies-bouldin', 'silhouette', 'knee'].

        This is simply a wrapper over sklearn.cluster.KMeans, and will accept any kwarg that the original
        class will accept.

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        If you pass remove=True, it will delete *.points files as they are split and saved into *.tracks files.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (list, optional): Specific list of agent ids to be processed. If None, processes all agents.
            tracks (list, optional): Specific list of track ids to be processed. If None, processes all tracks.
            n_clusters (int, list, range, optional): Int or list/range of numbers of clusters to try for determining the optimal number. Defaults to range(10).
            feature_cols (list of str, optional): List of column names to be used as features for clustering. Defaults to ['X', 'Y'].
            out_path (str, optional): Output path where the split tracks will be saved. If None, uses the current data path.
            ncores (int, optional): Number of cores to use for parallel processing. Defaults to 1.
            return_error (bool, optional): If True, returns error of the KMeans model for each number of clusters in n_clusters. Defaults to False.
            remove (bool, optional): Whether to remove the original *.points file after splitting. Defaults to True.
            desc (str, optional): Description of the operation. Defaults to 'Using KMeans clustering to split tracks'.
            optimal_method (str, optional): Method to use for determining the optimal number of clusters. Defaults to 'davies-bouldin'.
            **kwargs: Additional keyword arguments to pass to the KMeans clustering function.

        Returns:
            Depending on the value of return_error, either:
            - self: The Dataset instance, if return_error is False.
            - tuple: (self, error), where error is a DataFrame of errors for each cluster number in n_clusters, if return_error is True.
        """
        #set out path
        out_path = self._set_out_path(out_path)
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
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #process clustering in parallel
        error = utils.pool_caller(process.split_tracks_kmeans,
                                  (n_clusters, 
                                   feature_cols, 
                                   out_path, 
                                   return_error, 
                                   remove, 
                                   optimal_method,
                                   kwargs),
                                  pkl_groups,
                                  desc,
                                  ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
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
                            out_path=None,
                            ncores=1,
                            remove=True,
                            eps=0.5,
                            min_samples=2,
                            desc='Using DBSCAN clustering to split tracks',
                            **kwargs):
        """
        Splits tracks using the DBSCAN clustering algorithm based on specified feature columns.
        This method groups points into clusters based on their density, allowing for the identification
        of varying densities within the data to effectively split tracks.

        This is simply a wrapper over sklearn.cluster.DBSCAN, and will accept any kwargs the original
        class accepts.

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        If you pass remove=True, it will delete *.points files as they are split and saved into *.tracks files.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (optional): Specific list of agent IDs to be processed.
            tracks (optional): Specific list of track IDs to be processed.
            feature_cols (list of str, optional): Feature columns to be used for clustering. Defaults to ['X', 'Y'].
            out_path (str, optional): Output path for the split tracks. If None, uses the current data path. Defaults to None.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            remove (bool, optional): Whether to remove the original point files after splitting. Defaults to True.
            eps (float, optional): The maximum distance (normalized) between two samples for them to be considered as in the same neighborhood. Defaults to 0.5.
            min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 2.
            desc (str, optional): Description of the operation. Defaults to 'Using DBSCAN clustering to split tracks'.
            **kwargs: Additional keyword arguments passed to the sklearn.cluster.DBSCAN clustering algorithm.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #process clustering in parallel
        utils.pool_caller(process.split_tracks_dbscan,
                            (feature_cols, 
                            out_path, 
                            remove, 
                            eps, 
                            min_samples,
                            kwargs),
                            pkl_groups,
                            desc,
                            ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def repair_tracks_spatiotemporal(self, 
                                     agents=None,
                                     time=3600 * 12, #seconds, 12hrs
                                     distance=0.5, #same units as DataSet, default = 0.5deg = 55km
                                     ncores=1, 
                                     out_path=None,
                                     desc='Repairing tracks using spatiotemporal threshold'):
        """
        Repairs tracks by connecting disjoint segments based on spatiotemporal thresholds.
        This method is intended to identify and bridge gaps within tracks that are shorter than
        specified time and distance thresholds. It is useful for reconstructing tracks that may
        have been erroneously split due to missing data or other anomalies.

        This function assumes tracks have already been split by some other method.

        It's very similar to Dataset.split_overlapping_tracks_spatiotemporal, but you use it to fix
        erroneously split tracks.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (optional): Specific list of agent ids to be processed. If None, all agents are processed.
            time (int, optional): Time threshold in seconds for connecting disjoint track segments. Defaults to 43200 seconds (12 hours).
            distance (float, optional): Distance threshold in the same units as DataSet for connecting disjoint track segments. Defaults to 0.5, which is approximately 55km if data is geographic.
            ncores (int, optional): Number of processing cores to use. Defaults to 1.
            out_path (str, optional): Path where the repaired tracks should be saved. If None, tracks are saved in the current data path.
            desc (str, optional): Description of the operation being performed. Defaults to 'Repairing tracks using spatiotemporal threshold'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #process repairs in parallel
        utils.pool_caller(process.repair_tracks_spatiotemporal,
                          (time, 
                           distance, 
                           out_path),
                          pkl_files,
                          desc,
                          ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def remove_agents(self,
                      agents=None,
                      desc='Removing agents from Dataset',
                      ncores=1):
        """
        Removes specified agents from the Dataset. This operation is designed to
        selectively remove agents based on their identifiers, facilitating data management
        and cleaning processes. Get rid of data you don't need anymore!

        Args:
            agents (list, optional): List of agent ids to be removed. If None, it assumes all agents.
            desc (str, optional): Description of the operation. Defaults to 'Removing agents from Dataset'.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.

        Returns:
            self: The Dataset instance.
        """
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #remove the files in parallel
        utils.pool_caller(os.remove, 
                          (), 
                          pkl_files, 
                          desc, 
                          ncores)
        #refresh the status
        self.status = self._refresh_status()
        return self            
    
    def remove_tracks(self,
                      tracks=None,
                      ncores=1,
                      out_path=None,
                      desc='Removing tracks from Dataset'):
        """
        Removes specified tracks from the Dataset. This operation is designed to
        selectively remove tracks based on their identifiers, facilitating data management
        and cleaning processes. Get rid of data you don't need anymore!

        Args:
            tracks (list, optional): List of tracks ids to be removed. If None, it assumes all tracks.
            desc (str, optional): Description of the operation. Defaults to 'Removing tracks from Dataset'.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(None, tracks)))
        utils.pool_caller(process.remove_tracks,
                          (out_path,),
                          pkl_groups,
                          desc,
                          ncores)
        #refresh the status
        self.status = self._refresh_status()
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
        
    def get_track_splits(self, 
                         agents=None,
                         ncores=1,
                         desc='Getting track split data'):
        """
        Retrieves data related to the gaps between tracks. This function is useful for analyzing 
        how tracks have been divided spatiotemporally, and potentially isolating a list of
        tracks to rejoin.

        Args:
            agents (list, optional): List of agent ids for which track split data will be retrieved.
                                    If None, track split data for all agents will be retrieved.
            ncores (int, optional): Number of cores to use for parallel processing. Defaults to 1.
            desc (str, optional): Description of the operation. Defaults to 'Getting track split data'.

        Returns:
            DataFrame: A pandas DataFrame containing the track split data.
        """
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
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
        """
        Refreshes the metadata associated with the Dataset. This involves updating the
        dataset.db, agent.db, and track.db files stored in the Dataset.data_path folder.
        If these files do not exist (i.e. Dataset.refresh_meta has never been run yet),
        it will created them.

        You only need to do this if you want the .db files to be updated, which is only
        necessary if you need an updated output for Dataset.meta, Dataset.agents, or Dataset.tracks.
        
        Passing agents_only=True should only be used when points have been grouped into *.points files
        but have not yet been split into tracks. This will provide metadata on the agents that 
        can be useful for doing categorical splitting. For example, splitting AIS data for cargo vessels
        differently than fishing vessels.

        Args:
            agents_only (bool, optional): If True, the refresh operation will be limited to metadata
                                        pertaining to agents only. Defaults to False, meaning all metadata
                                        will be refreshed.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            desc (str, optional): Description of the operation. Defaults to 'Refreshing metadata'.

        Returns:
            self: The Dataset instance.
        """
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
                         meta_cols,
                         agents=None,
                         ncores=1,
                         desc='Making meta mappers'):
        """
        Creates a metadata mapper dictionary based on the meta_cols provided. This mapper can be used to
        translate or map metadata values, potentially simplifying data analysis and manipulation
        tasks.

        Args:
            meta_cols (list): The meta_cols used to create the meta mapper.
            agents (list, optional): List of agent ids for which the meta mapper will be made.
                                   If None, the mapper will be created using all agents.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            desc (str, optional): Description of the operation. Defaults to 'Making meta mappers'.

        Returns:
            dict: A dictionary where keys are the names of the data columns specified in `meta_cols`, and
                the values are the created data mappers.
        """
        #must be list of data columns
        msg = 'meta_cols must be a list of meta columns'
        assert isinstance(meta_cols, list) and len(meta_cols) > 0, msg
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #read data from agents in parallel
        _out =  utils.pool_caller(maps.make_meta_mapper,
                                 (meta_cols,),
                                 pkl_files,
                                 desc,
                                 ncores)
        #reduce to unique dictionary
        out = utils.flatten_dict_unique(_out)
        #turn into blank mapper
        mappers = [{k:None for k in out[key]} for key in meta_cols]
        #if single mapper just return it
        if len(mappers) == 1:
            return mappers[0]
        #otherwise nested dict
        else:
            return dict(zip(meta_cols, mappers))

    def drop_meta(self, 
                  meta_cols,
                  agents=None,
                  out_path=None,
                  ncores=1,
                  desc='Dropping meta from agents'):
        """
        Drops specified metadata from agents. This function allows for selective removal of metadata
        associated with certain agents, potentially for data cleaning, or to reduce
        dataset size.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            meta_cols (list): The meta_cols to be dropped.
            agents (list, optional): List of agent ids from which the metadata will be dropped. If None,
                                    the operation is applied to all agents.
            out_path (str, optional): Output path where the modified dataset should be saved. If None,
                                    defaults to current data_path.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            desc (str, optional): Description of the operation. Defaults to 'Dropping meta from agents'.

        Returns:
            self: The Dataset instance.
        """
        #must be list of data columns
        msg = 'meta_cols must be a list of meta columns'
        assert isinstance(meta_cols, list) and len(meta_cols) > 0, msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #delete data from agents in parallel
        utils.pool_caller(maps.drop_agent_meta,
                        (meta_cols, out_path),
                        pkl_files,
                        desc,
                        ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
        
    def map_meta(self,
                meta_cols, 
                out_meta_cols,
                meta_mappers,
                agents=None,
                out_path=None,
                ncores=1,
                drop=False,
                fill=None,
                desc='Mapping metadata to agent'):
        """
        Applies a mapping function to transform metadata of selected agents. This can be used to
        normalize, categorize, or otherwise process metadata fields for consistency, analysis,
        or data cleaning purposes. The transformation is defined by the `mapper` dictionary which specifies
        how input metadata values are mapped to output values.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            meta_cols (list): The meta columns to be transformed.
            out_meta_cols (list): The name of the mapped meta columns to be added.
            meta_mappers (dict): Dictionary that specifies how each input value
                    is transformed to an output value.
            agents (list, optional): A list of agent IDs indicating which agents' metadata should be transformed.
                                    If None, the operation is applied to all agents.
            out_path (str, optional): The file path where the dataset with the transformed metadata should be saved.
                                    If None, the operation uses the current data_path.
            ncores (int, optional): The number of processing cores to use for the operation. Defaults to 1.
            drop (bool, optional): Whether to remove the input metadata after transformation. Defaults to False,
                                    indicating that the input metadata will be retained unless specified otherwise.
            fill (optional): A value to fill in for missing or undefined mappings.
            desc (str, optional): A brief description of the metadata mapping operation. Defaults to 'Mapping metadata to agent'.

        Returns:
            self: The Dataset instance.
        """
        #must be list and proper dict
        msg = f'meta_mappers missing one or all of keys: {meta_cols}'
        assert all([i in meta_mappers.keys() for i in meta_cols]), msg
        msg = 'out_meta_cols is not the same length as meta_cols'
        assert len(meta_cols) == len(out_meta_cols), msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #map metadata to agents in parallel
        utils.pool_caller(maps.map_agent_meta,
                          (meta_cols, out_meta_cols, meta_mappers, out_path, drop, fill),
                          pkl_files,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def make_data_mapper(self,
                         data_cols,
                         agents=None,
                         tracks=None,
                         ncores=1,
                         desc='Making data mappers'):
        """
        Creates a mapping for data columns to facilitate data transformation. This function
        can be particularly useful for preparing data for machine learning models, data visualization,
        or statistical analysis by mapping raw data into a more useful format, or to just simply
        clean or categorize data into a new, easier field to work with.

        Args:
            data_cols (list): The names of the data columns for which the mappers will
                                be created.
            agents (list, optional): A list of agent IDs for which the data mappers will be specifically
                                    created. If None, mappers will be created considering all agents.
            tracks (list, optional): A list of track IDs for which the data mappers will be specifically
                                    created. If None, it will used all of the tracks.
            ncores (int, optional): The number of processing cores to use for creating the data mappers.
                                    Defaults to 1.
            desc (str, optional): A brief description of the purpose or the process of making data mappers.
                                Defaults to 'Making data mappers'.

        Returns:
            dict: A dictionary where keys are the names of the data columns specified in `data_cols`, and
                the values are the created data mappers.
        """
        #must be list of data columns
        msg = 'data_cols must be a list of data columns'
        assert isinstance(data_cols, list) and len(data_cols) > 0, msg
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #read data from agents in parallel
        _out =  utils.pool_caller(maps.make_data_mapper,
                                 (data_cols,),
                                 pkl_groups,
                                 desc,
                                 ncores)
        #reduce to unique dictionary
        out = utils.flatten_dict_unique(_out)
        #turn into blank mapper
        mappers = [{k:None for k in out[key]} for key in data_cols]
        #if single mapper just return it
        if len(mappers) == 1:
            return mappers[0]
        else:
            return dict(zip(data_cols, mappers))
    
    def drop_data(self, 
                  data_cols,
                  agents=None,
                  out_path=None,
                  ncores=1,
                  desc='Dropping dynamic data from agents'):
        """
        Drops specified dynamic data from agents. This function allows for selective removal of data
        associated with certain agents, potentially for data cleaning, or to reduce
        dataset size.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            data_cols (list): The data_cols to be dropped.
            agents (list, optional): List of agent ids from which the metadata will be dropped. If None,
                                    the operation is applied to all agents.
            out_path (str, optional): Output path where the modified dataset should be saved. If None,
                                    defaults to current data_path.
            ncores (int, optional): Number of cores to use for processing. Defaults to 1.
            desc (str, optional): Description of the operation. Defaults to 'Dropping dynamic data from agents'.

        Returns:
            self: The Dataset instance.
        """
        #must be list of data columns
        msg = 'data_cols must be a list of data columns'
        assert isinstance(data_cols, list) and len(data_cols) > 0, msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #delete data from agents in parallel
        utils.pool_caller(maps.drop_agent_data,
                        (data_cols, out_path),
                        pkl_files,
                        desc,
                        ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def map_data(self,
                data_cols=[],
                out_data_cols=[],
                data_mappers={},
                agents=None,
                tracks=None,
                out_path=None,
                ncores=1,
                drop=False,
                fill=None,
                desc='Mapping agent dynamic data'):
        """
        Applies a mapping function to transform dynamic data of selected agents/tracks. This can be used to
        normalize, categorize, or otherwise process data fields for consistency, analysis,
        or data cleaning purposes. The transformation is defined by the mapper dictionary which specifies
        how input data values are mapped to output values.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            data_cols (list): The data columns to be transformed.
            out_data_cols (list): The name of the mapped data columns to be added.
            data_mappers (dict): Dictionary that specifies how each input value
                    is transformed to an output value.
            agents (list, optional): A list of agent IDs indicating which agents' data should be transformed.
                                    If None, the operation is applied to all agents.
            out_path (str, optional): The file path where the dataset with the transformed data should be saved.
                                    If None, the operation uses the current data_path.
            ncores (int, optional): The number of processing cores to use for the operation. Defaults to 1.
            drop (bool, optional): Whether to remove the input data column(s) after transformation. Defaults to False.
            fill (optional): A value to fill in for missing or undefined mappings.
            desc (str, optional): A brief description of the data mapping operation. Defaults to 'Mapping agent dynamic data'.

        Returns:
            self: The Dataset instance.
        """
        #must be list and proper dict
        msg = f'data_mappers missing one or all of keys: {data_cols}'
        assert all([i in data_mappers.keys() for i in data_cols]), msg
        msg = 'out_data_cols is not the same length as data_cols'
        assert len(data_cols) == len(out_data_cols), msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #map agent data in parallel
        utils.pool_caller(maps.map_agent_data,
                          (data_cols, out_data_cols, data_mappers, out_path, drop, fill),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self

    def map_data_to_codes(self,
                          data_cols,
                          data_mappers,
                          agents=None,
                          tracks=None,
                          out_path=None,
                          ncores=1,
                          drop=False,
                          fill=-1,
                          desc='Mapping agent dynamic data to coded boolean arrays'):
        """
        Applies a mapping function to transform dynamic data of selected agents/tracks into boolean "Code"
        columns.

        For this to work, the values in the data_mappers dictionaries must all be integer values.
        The algorithm will make a coded boolean column for each integer in the dictionary, True
        where it is that value and False where it isn't.

        Each boolean column will be called "Code{i}" where i corresponds to one of the unique integers
        in the dictionary values.
          
        This can be used to encode information into boolean code columns that will be available
        in the Dataset.agents and Dataset.tracks attributes, which could make for faster and more
        efficient querying.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            data_cols (list): The data columns to be transformed.
            data_mappers (dict): Dictionary that specifies how each input value
                    is transformed to an output value. All values must be integers.
            agents (list, optional): A list of agent IDs indicating which agents' data should be transformed.
                                    If None, the operation is applied to all agents.
            out_path (str, optional): The file path where the dataset with the transformed data should be saved.
                                    If None, the operation uses the current data_path.
            ncores (int, optional): The number of processing cores to use for the operation. Defaults to 1.
            drop (bool, optional): Whether to remove the input data column(s) after transformation. Defaults to False.
            fill (optional): A value to fill in for missing or undefined mappings.
            desc (str, optional): A brief description of the data mapping operation. Defaults to 'Mapping agent dynamic data to coded boolean arrays'.

        Returns:
            self: The Dataset instance.
        """
        #must be list and proper dict
        msg = 'data_cols must be a list of data columns'
        assert len(data_cols)>0 and isinstance(data_cols, list), msg
        msg = f'data_mappers missing one or all of keys: {data_cols}'
        assert all([i in data_mappers.keys() for i in data_cols]), msg
        #assert all mapper values are integer
        msg = 'All mapper values must be integer codes'
        for m in data_mappers.values():
            assert all([isinstance(v, int) for v in m.values()]), msg
        #check fill value
        assert isinstance(fill, int), 'fill value must be integer'
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #map agent data in parallel
        utils.pool_caller(maps.map_agent_data_to_codes,
                          (data_cols, data_mappers, out_path, drop, fill),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self

    ############################################################################
    #FETCHERS
    ############################################################################

    #make deepcopy
    def copy(self):
        """
        Creates a deepcopy of the input Dataset.

        Returns:
            Dataset: Deepcopy of the input Dataset.
        """
        return deepcopy(self)
    
    #get one agent
    def get_agent(self, agent):
        """
        Reads and returns one single agent file, specified by Agent ID.

        Args:
            agent: Unique identifier (Agent ID) of the agent to read.

        Returns:
            Dataset: The requested agent.
        """
        return utils.collect_agent_pkls(self._file_mapper['Agents'][agent])
    
    #get specific agents
    def get_agents(self, agents, ncores=1):
        """
        Reads and returns a list of agent files, specified by Agent IDs.

        Args:
            agents: Unique identifiers (Agent IDs) of the agents to read.

        Returns:
            list: List of the requested agents.
        """
        agent_files = self._file_mapper['Agents']
        files = [agent_files[a] for a in agents]
        #process in parallel
        with mp.Pool(ncores) as pool:
            out = pool.map(utils.collect_agent_pkls, tqdm(files, 
                                                          desc=GREEN+'Getting agents'+ENDC,
                                                          total=len(files),
                                                          colour='GREEN'))
        return out
    
    #get one track
    def get_track(self, track):
        """
        Reads and returns one single track, specified by Track ID.

        Args:
            track: Unique identifier (Track ID) of the track to read.

        Returns:
            Dataset: The requested track.
        """
        aid, tid = track.rsplit('_',1)
        file = self._file_mapper['Tracks'][aid]
        a = utils.read_pkl(file)
        return a.tracks[tid]
            
    #get specific tracks
    def get_tracks(self, tracks, ncores=1):
        """
        Reads and returns a list of tracks, specified by Track IDs.

        Args:
            tracks: Unique identifiers (Track IDs) of the tracks to read.

        Returns:
            list: List of the requested tracks.
        """
        aids = list(map(lambda x: x.rsplit('_', 1)[0], tracks))
        files = [self._file_mapper['Tracks'][a] for a in aids]
        #process in parallel
        with mp.Pool(ncores) as pool:
            out = pool.map(utils.read_pkl, tqdm(files, 
                                                desc=GREEN+'Getting tracks'+ENDC,
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
        """
        Converts requested agents/tracks into a GeoDataFrame (gdf). 

        By default, one LineString is created for each track. You can pass segments=True to "explode"
        these LineStrings and create N number of LineString segments for each track that has N+1 points.

        You can also pass an integer code value to only convert the data where that code value is true. In
        the default case, this would result in one MultiLineString for each track. In the segments=True case,
        the segments where this condition is false would simply be dropped.

        The method kwarg must be one of ['forward', 'middle', 'backward'] and corresponds to the fill method
        for determining values of data at segments. If 'forward', each segment has the value of the start point,
        for 'backward' it uses the end point, and for 'middle' it takes an average.

        Args:
            agents (list, optional): A list of agent IDs for which tracks will be converted. If None, the
                                    conversion process considers all agents available in the dataset.
            tracks (list, optional): A list of track IDs to be converted. This parameter allows for the
                                    selection of specific tracks for conversion. If None, all tracks related
                                    to the specified agents (or all agents if agents is None) will be converted.
            code (int, optional): A code column to apply to the conversion. If provided, only portions of the tracks
                                 where this code is True will be converted. If segments=True, it will only return
                                 segments where this code is True.
            ncores (int, optional): The number of processing cores to use for the conversion process. Defaults to 1.
            segments (bool, optional): If True, tracks will be converted into individual segments rather than a single
                                    line. Defaults to False.
            method (str, optional): Specifies the fill/interpolation method for dynamic data values to be evaluated
                                   at segments when segments=True. Must be one of ['forward', 'middle', 'backward'].
                                   Defaults to middle.
            desc (str, optional): A brief description of the conversion process or its purpose. Defaults to
                                'Converting tracks to GeoDataFrame'.

        Returns:
            GeoDataFrame: A GeoPandas GeoDataFrame containing the converted track data.
        """
        #change desc
        if segments:
            desc = 'Converting track segments to GeoDataFrame'
        #assert method is valid
        if segments:
            msg = 'method must be backward, middle, or forward'
            assert method in ['backward','middle','forward'], msg
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))  
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
        """
        Converts agent/track data into a pandas DataFrame. This method allows for simple integration
        of pandas into data processing workflows, and leveraging the pandas framework and built-in tools.

        This function converts all of the requested agents/tracks into a DataFrame containing the points
        making up this data. Each row/point gets tagged with the Agent ID, Track ID, and Ping ID for
        recordkeeping purposes.

        You can also pass an integer code value to only convert the data where that code value is True.

        Args:
            agents (list, optional): A list of agent IDs whose data is to be converted into a DataFrame.
                                    If None, the function will attempt to convert data for all agents
                                    available in the dataset.
            tracks (list, optional): A list of specific track IDs to be converted. This is useful for focusing
                                    on particular tracks of interest. If None, the conversion process includes
                                    tracks related to the specified agents or all tracks if no agents are specified.
            code (int, optional): A code column to apply to the conversion. If provided, only portions of the tracks
                                 where this code is True will be converted.
            ncores (int, optional): The number of processing cores to use for the conversion operation. Defaults
                                    to 1.
            desc (str, optional): A brief description of the operation or its purpose. Defaults to 'Converting tracks
                                to DataFrame'.

        Returns:
            DataFrame: A pandas DataFrame containing the converted data.
        """
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))  
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
        """
        Converts the dataset for specified agents into a Dask Bag, enabling parallel processing and analysis
        of large datasets that do not fit into memory. Dask Bags are well-suited for working with unstructured
        data or data that can be processed in a sequence of operations.

        This can be used to leverage Dask for parallel processing of any custom functionality.

        Each element in the bag is a trackio.Agent. The tracks for each agent are accessible through
        the agent.tracks attribute, which is a dictionary containing numbered tracks (i.e. T0, T1, T2, etc.).

        Args:
            agents (list, optional): A list of agent IDs whose data is to be converted into a Dask Bag.
                                    If None, the function will convert data for all agents available in
                                    the dataset.

        Returns:
            dask.bag.Bag: A Dask Bag object where each element in the bag is a trackio.Agent.
        """
        #get the files to process
        pkl_files, _ = self._get_files_tracks_to_process(agents, None)
        pkl_files = utils.flatten(pkl_files)
        #create bag
        bag = db.from_sequence(pkl_files)
        return bag.map(utils.read_pkl)

    ############################################################################
    #GEOMETRIC
    ############################################################################
    
    def reproject_crs(self, 
                      crs,
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      out_path=None,
                      desc='Reprojecting CRS'):
        """
        Reprojects the coordinate reference system (CRS) of the dataset's geographic data to a new CRS.

        The target CRS can be a EPSG code, WKT string, pyproj.CRS object, etc.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            crs: The target coordinate reference system to which the dataset will be reprojected.
            agents (list, optional): A list of agent IDs whose data is to be reprojected. If None, the function
                                    aims to reproject data for all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to reproject. If None, the operation
                                    applies to tracks related to the specified agents or all tracks if no agents
                                    are specified.
            ncores (int, optional): The number of processing cores to use for the reprojection operation. Defaults
                                    to 1.
            out_path (str, optional): The file path where the dataset with the reprojected geographic data should be
                                    saved. If None, it defaults to the current data_path.
            desc (str, optional): A brief description of the reprojection operation. Defaults to 'Reprojecting CRS'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #get crs
        crs1 = CRS(self.meta['CRS'])
        crs2 = CRS(crs)
        #if already same crs
        if crs1.equals(crs2):
            pass
        #if different crs
        else:
            #make transformer once
            transformer = Transformer.from_crs(crs1, crs2, always_xy=True)
            #reproject in parallel
            utils.pool_caller(geometry.reproject_crs,
                                (transformer, out_path),
                                pkl_groups,
                                desc, 
                                ncores)
            #reset/write the meta
            meta = self.meta.copy()
            meta['CRS'] = crs2
            meta['X'] = crs2.axis_info[0].unit_name
            meta['Y'] = crs2.axis_info[1].unit_name
            #update meta
            self = self._update_meta(out_path, meta)
        return self
         
    def resample_spacing(self, 
                         spacing, 
                         agents=None,
                         tracks=None, 
                         ncores=1, 
                         out_path=None,
                         desc='Resampling track spacing'):
        """
        Resamples the points of tracks to a specified spacing. This is useful for standardizing
        the distance between consecutive points in track data, facilitating analyses that require uniform
        spatial intervals. The resampling process can either interpolate or decimate points to achieve the
        desired spacing.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.


        Args:
            spacing: The target spacing between points within tracks. The unit of measurement for spacing
                    depends on the coordinate reference system of the dataset (e.g., meters in a projected
                    CRS or degrees in a geographic CRS).
            agents (list, optional): A list of agent IDs to apply the resampling to. If None, the function
                                    will apply the resampling to tracks associated with all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to resample. This parameter allows for
                                    selective resampling of certain tracks. If None, all tracks will be
                                    considered.
            ncores (int, optional): The number of processing cores to use for the resampling operation. Defaults
                                    to 1.
            out_path (str, optional): The file path where the dataset with the resampled tracks should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the resampling operation. Defaults to 'Resampling track spacing'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.resample_spacing,
                          (spacing, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
        
    def resample_time(self, 
                      seconds, 
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      out_path=None,
                      desc='Resampling track timing'):
        """
        Resamples the points of tracks to a specified temporal spacing. This is useful for standardizing
        the timing between consecutive points in track data, facilitating analyses that require uniform
        temporal intervals. The resampling process can either interpolate or decimate points to achieve the
        desired temporal spacing.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            seconds: The target temporal spacing between points within tracks, in seconds.
            agents (list, optional): A list of agent IDs to apply the resampling to. If None, the function
                                    will apply the resampling to tracks associated with all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to resample. This parameter allows for
                                    selective resampling of certain tracks. If None, all tracks will be
                                    considered.
            ncores (int, optional): The number of processing cores to use for the resampling operation. Defaults
                                    to 1.
            out_path (str, optional): The file path where the dataset with the resampled tracks should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the resampling operation. Defaults to 'Resampling track timing'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.resample_time,
                          (seconds, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def resample_time_global(self, 
                             time, 
                             agents=None,
                             tracks=None, 
                             ncores=1, 
                             out_path=None,
                             desc='Resampling tracks to global time axis'):
        """
        Resamples the points of tracks to a global time axis.

        This is useful for reducing data to the same time axis for analyses which may require (or benefit)
        from this behaviour. One example would be the Dataset.proximities functionality, which calculates the closest
        point of approach (CPA); this would not work if the timestamps were not common between the tracks.

        This may also be useful for standardizing data for ML approaches.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            time (list): The target time axis to interpolate tracks to. Must be a list of datetime objects, or something
                        that can be converted using pandas.to_datetime.
            agents (list, optional): A list of agent IDs to apply the resampling to. If None, the function
                                    will apply the resampling to tracks associated with all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to resample. This parameter allows for
                                    selective resampling of certain tracks. If None, all tracks will be
                                    considered.
            ncores (int, optional): The number of processing cores to use for the resampling operation. Defaults
                                    to 1.
            out_path (str, optional): The file path where the dataset with the resampled tracks should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the resampling operation. Defaults to 'Resampling tracks to global time axis'.

        Returns:
            self: The Dataset instance.
        """
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
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.resample_time_global,
                          (_time, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self    

    def compute_coursing(self, 
                         agents=None,
                         tracks=None, 
                         ncores=1, 
                         out_path=None,
                         method='middle',
                         desc='Computing coursing'):
        """
        Computes the coursing of tracks based on the direction travelled between points. This gets added
        as a "Coursing" column in the track data.

        The method kwarg must be one of ['forward', 'middle', 'backward'] and corresponds to which pairs of
        points to use for determining coursing values, and how to fill the remaining point. 

        If 'forward', the coursing at each point is calculated by the direction from the current
        point to the next point. The first point on the track then gets filled with the coursing of the second point.

        If 'backward', it uses the current point and the previous point, and the last point gets filled with
        the coursing value of the second last point.

        If 'middle', it uses an average of both and no filling is required.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.
        
        Args:
            agents (list, optional): A list of agent IDs for which coursing will be computed. If None, coursing
                                    will be computed for all agents available in the dataset.
            tracks (list, optional): A list of specific track IDs for which to compute coursing. This allows
                                    focusing on particular tracks of interest. If None, and agents are specified,
                                    coursing is computed for all tracks associated with the specified agents.
                                    If both are None, coursing is computed for all tracks in the dataset.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the results of the coursing computation should be saved.
                                    If None, it uses the current data_path.
            method (str, optional): The method used to compute coursing, must be one of ['forward', 'middle', 'backward'].
            desc (str, optional): A brief description of the coursing computation operation. Defaults to 'Computing coursing'.

        Returns:
            self: The Dataset instance.
        """
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_coursing,
                          (method, self.meta['CRS'], out_path),
                          pkl_groups,
                          desc,
                          ncores)       
        #update meta
        meta = self.meta.copy()
        meta['Coursing'] = 'degrees'
        self = self._update_meta(out_path, meta)
        return self
    
    def compute_turning_rate(self, 
                             agents=None,
                             tracks=None, 
                             ncores=1, 
                             out_path=None,
                             method='middle',
                             desc='Computing turning rate'):
        """
        Computes the turning rate of tracks based on the change in coursing between points. This gets added
        as a "Turning Rate" column in the track data.

        This method assumes that there is accurate coursing data available for the tracks. If this is
        not the case, you should run the Dataset.compute_coursing function first.

        The method kwarg must be one of ['forward', 'middle', 'backward'] and corresponds to which pairs of
        points to use for determining turning rate values, and how to fill the remaining point. 

        If 'forward', the turning rate at each point is calculated by the change in coursing from the current
        point to the next point. The first point on the track then gets filled with the value of the second point.

        If 'backward', it uses the current point and the previous point, and the last point gets filled with
        the value of the second last point.

        If 'middle', it uses an average of both and no filling is required.
        
        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (list, optional): A list of agent IDs for which turning rate will be computed. If None, turning rate
                                    will be computed for all agents available in the dataset.
            tracks (list, optional): A list of specific track IDs for which to compute turning rate. This allows
                                    focusing on particular tracks of interest. If None, and agents are specified,
                                    turning rate is computed for all tracks associated with the specified agents.
                                    If both are None, turning rate is computed for all tracks in the dataset.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the results of the turning rate computation should be saved.
                                    If None, it uses the current data_path.
            method (str, optional): The method used to compute turning rate, must be one of ['forward', 'middle', 'backward'].
            desc (str, optional): A brief description of the turning rate computation operation. Defaults to 'Computing turning rate'.

        Returns:
            self: The Dataset instance.
        """
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_turning_rate,
                          (method, out_path),
                          pkl_groups,
                          desc,
                          ncores)       
        #update meta
        meta = self.meta.copy()
        meta['Turning Rate'] = 'degrees/sec'
        self = self._update_meta(out_path, meta)
        return self

    def compute_speed(self, 
                      agents=None,
                      tracks=None, 
                      ncores=1, 
                      out_path=None,
                      method='middle',
                      desc='Computing speed'):
        """
        Computes the speed of tracks based on the distance/time between points. This gets added
        as a "Speed" column in the track data.

        The method kwarg must be one of ['forward', 'middle', 'backward'] and corresponds to which pairs of
        points to use for determining speed values, and how to fill the remaining point. 

        If 'forward', the speed at each point is calculated by the change in distance/time from the current
        point to the next point. The first point on the track then gets filled with the value of the second point.

        If 'backward', it uses the current point and the previous point, and the last point gets filled with
        the value of the second last point.

        If 'middle', it uses an average of both and no filling is required.
        
        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (list, optional): A list of agent IDs for which speed will be computed. If None, speed
                                    will be computed for all agents available in the dataset.
            tracks (list, optional): A list of specific track IDs for which to compute speed. This allows
                                    focusing on particular tracks of interest. If None, and agents are specified,
                                    speed is computed for all tracks associated with the specified agents.
                                    If both are None, speed is computed for all tracks in the dataset.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the results of the speed computation should be saved.
                                    If None, it uses the current data_path.
            method (str, optional): The method used to compute speed, must be one of ['forward', 'middle', 'backward'].
            desc (str, optional): A brief description of the speed computation operation. Defaults to 'Computing speed'.

        Returns:
            self: The Dataset instance.
        """
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_speed,
                          (method, out_path),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Speed'] = f"{meta['X']}/second"
        self = self._update_meta(out_path, meta)
        return self

    def compute_acceleration(self, 
                             agents=None,
                             tracks=None, 
                             ncores=1, 
                             out_path=None,
                             method='middle',
                             desc='Computing acceleration'):
        """
        Computes the acceleration of tracks based on the distance/time between points. This gets added
        as an "Acceleration" column in the track data.

        This method assumes that there is accurate speed data available for the tracks. If this is
        not the case, you should run the Dataset.compute_speed method first.

        The method kwarg must be one of ['forward', 'middle', 'backward'] and corresponds to which pairs of
        points to use for determining acceleration values, and how to fill the remaining point. 

        If 'forward', the acceleration at each point is calculated by the change in speed from the current
        point to the next point. The first point on the track then gets filled with the value of the second point.

        If 'backward', it uses the current point and the previous point, and the last point gets filled with
        the value of the second last point.

        If 'middle', it uses an average of both and no filling is required.
        
        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (list, optional): A list of agent IDs for which acceleration will be computed. If None, acceleration
                                    will be computed for all agents available in the dataset.
            tracks (list, optional): A list of specific track IDs for which to compute acceleration. This allows
                                    focusing on particular tracks of interest. If None, and agents are specified,
                                    acceleration is computed for all tracks associated with the specified agents.
                                    If both are None, acceleration is computed for all tracks in the dataset.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the results of the acceleration computation should be saved.
                                    If None, it uses the current data_path.
            method (str, optional): The method used to compute acceleration, must be one of ['forward', 'middle', 'backward'].
            desc (str, optional): A brief description of the acceleration computation operation. Defaults to 'Computing acceleration'.

        Returns:
            self: The Dataset instance.
        """
        #assert method
        msg = 'method must be backward, middle, or forward'
        assert method in ['backward','middle','forward'], msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_acceleration,
                          (out_path, method),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Acceleration'] = f"{meta['Speed']}/second"
        self = self._update_meta(out_path, meta)
        return self

    def compute_distance_travelled(self, 
                                   relative=False,
                                   agents=None,
                                   tracks=None, 
                                   ncores=1, 
                                   out_path=None,
                                   desc='Computing distance travelled along tracks'):
        """
        Computes the total distance travelled along each track. This gets added
        as a "Distance Travelled" column in the track data.
        
        The computation can be done in an absolute sense which captures the true distance travelled
        along the track. Or, it can be done relatively where every track is normalized to travel from 0-1.

        Args:
            relative (bool, optional): If True, result is normalized between 0-1 for all tracks.
            agents (list, optional): A list of agent IDs for which the distance travelled will be computed.
                                    If None, the computation is applied across all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which the distance travelled will be computed.
                                    This allows for focusing on particular tracks of interest. If None, the computation
                                    applies to tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the dataset with the computed distances should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the operation. Defaults to 'Computing distance travelled
                                along tracks'.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_distance_travelled,
                          (out_path, relative),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Distance Travelled'] = meta['X']
        self = self._update_meta(out_path, meta)
        return self

    def compute_radius_of_curvature(self, 
                                   agents=None,
                                   tracks=None, 
                                   ncores=1, 
                                   out_path=None,
                                   desc='Computing radius of curvature'):
        """
        Computes the radius of curvature for each point along tracks. This gets added
        as a "Radius of Curvature" column in the track data.

        This is very useful for isolating parts of tracks where agents are turning.
        
        This algorithm works by looking at points along each track, along with the 2 neighbouring points. 
        These 3 points are then used to construct a circle, and the radius of this circle is the Radius of Curvature
        at the middle point. Naturally, this results in a nan value for the first and last points of the tracks, 
        since they only have 1 neighbour.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            agents (list, optional): A list of agent IDs for which the radius of curvature will be computed.
                                    If None, the computation is applied across all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which the radius of curvature will be computed.
                                    This allows for focusing on particular tracks of interest. If None, the computation
                                    applies to tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the dataset with the computed radius of curvature should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the operation. Defaults to 'Computing radius of curvature'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_radius_of_curvature,
                          (out_path,),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Radius of Curvature'] = meta['X']
        self = self._update_meta(out_path, meta)
        return self

    def compute_sinuosity(self, 
                          agents=None,
                          tracks=None, 
                          ncores=1, 
                          out_path=None,
                          window=3,
                          desc='Computing sinuosity'):
        """
        Computes the sinuosity for each point along tracks. This gets added as a "Sinuosity" column in the track data.
        
        This algorithm works by looking at a central middle point, as well as a user-defined window of points centered
        at this point. The sinuosity is then calculated as the ratio between total length and effective length
        (distance between start and end points) for the window of points. Naturally, this will result in nan values
        at the beginning and end of the track depending on how large the window is.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            window (int): The size of the window to consider at each point. This window is centered at each point (i.e. (window-1)/2 on either side).
                          Defaults to 3.
            agents (list, optional): A list of agent IDs for which the sinuosity will be computed.
                                    If None, the computation is applied across all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which the sinuosity will be computed.
                                    This allows for focusing on particular tracks of interest. If None, the computation
                                    applies to tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the computation. Defaults to 1.
            out_path (str, optional): The file path where the dataset with the computed sinuosity should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the operation. Defaults to 'Computing sinuosity'.

        Returns:
            self: The Dataset instance.
        """
        #assert odd window number to be centered on each point
        assert window%2 > 0, 'window must be an odd number'
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #recompute in parallel
        utils.pool_caller(geometry.compute_sinuosity,
                          (out_path, window),
                          pkl_groups,
                          desc,
                          ncores)  
        #update meta
        meta = self.meta.copy()
        meta['Sinuosity'] = 'non-dimensional'
        self = self._update_meta(out_path, meta)
        return self

    def smooth_corners(self, 
                       refinements=2,
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       out_path=None,
                       desc='Smoothing sharp corners'):
        """
        Smooths sharp corners/turns in tracks using a iterative weighted averaging technique.

        This is useful for smoothing jagged corners/turns in data where the quality of the data
        around turns is important, or simply where the resolution of data around turns is poor and you
        want to smooth the track to be more realistic.

        Here, linear interpolation is used to fill any dynamic data at new points.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            refinments (int): The number of smoothing iterations to perform. Defaults to 2.
            agents (list, optional): A list of agent IDs for which the operation will be performed.
                                    If None, the operation is applied across all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which the the operation will be performed.
                                    This allows for focusing on particular tracks of interest. If None, the operation
                                    applies to tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the opreation. Defaults to 1.
            out_path (str, optional): The file path where the resulting dataset should be saved.
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the operation. Defaults to 'Smoothing sharp corners'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.smooth_corners,
                          (refinements, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
            
    def decimate_tracks(self, 
                        epsilon=1,
                        agents=None,
                        tracks=None, 
                        ncores=1, 
                        out_path=None,
                        desc='Decimating tracks'):
        """
        Decimates track geometry using the Douglas-Peucker algorithm. 

        https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        
        Decimation is very useful for reducing the size of the dataset, while still
        maintaining an appropriate level of detail.

        Here, epsilon is in the units of the dataset coordinate system. I.e. degrees for geographic,
        meters for UTM, etc.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            epsilon (float, optional): The tolerance for decimation, determining the minimum distance 
                                    a point must have from a line segment connecting adjacent points 
                                    to be retained. Defaults to 1, with units depending on the CRS 
                                    of the track data (e.g., meters or degrees).
            agents (list, optional): A list of agent IDs for which tracks will be decimated. If None, 
                                    the decimation process is applied to tracks associated with all 
                                    agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be decimated. This allows for 
                                    selective decimation of tracks. If None, the operation applies 
                                    to tracks related to the specified agents or all tracks if no 
                                    agents are specified.
            ncores (int, optional): The number of processing cores to use for the decimation operation. 
                                    Defaults to 1.
            out_path (str, optional): The file path where the dataset with decimated tracks should be saved. 
                                    If None, it assumes the current data_path.
            desc (str, optional): A brief description of the decimation operation. Defaults to 'Decimating tracks'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.decimate_tracks,
                          (epsilon, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
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
                              out_path=None,
                              inplace=False,
                              desc='Extracting characteristic tracks'):
        """
        Based on "Spatial Generalization and Aggregation of Massive Movement Data", Adrienko & Adrienko (2010).

        Paper here:
        http://geoanalytics.net/and/papers/tvcg11.pdf

        Identifies and extracts characteristic tracks from existing tracks.

        This method is useful for analyzing macroscopic behavior or movement patterns within track data, allowing for the 
        identification of significant activities or navigation patterns based on parameters like stop duration, 
        turn angles, and travel distances.

        This method is also very useful for simplifying tracks to reduce dataset size, while maintaining an
        appropriate level of detail compared to the original data.

        If you pass inplace=True, the tracks will actually get reduced to their characteristic points. If you
        pass inplace=False, a boolean "Charactersitic" column will be added to the track dataframes that is
        True at characteristic points.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Here all spatial/speed kwargs are in the CRS units. I.e. degrees for geographic, meters for UTM, etc.

        Args:
            stop_threshold (float, optional): The speed threshold below which a movement is considered a stop. Defaults to 0.15.
            turn_threshold (float, optional): The minimum change in direction (in degrees) considered a significant 
                                                turn. Defaults to 22.5 degrees.
            min_distance (float, optional): The minimum allowable distance between characteristic points. Defaults to 500.
            max_distance (float, optional): The maximum allowable distance between characteristic points. Defaults to 20000.
            min_stop_duration (int, optional): The minimum duration (in seconds) of a stop to be considered significant. 
                                                Defaults to 1800 seconds (30 minutes).
            agents (list, optional): A list of agent IDs to be analyzed. If None, the analysis is applied across all 
                                        agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be analyzed. This allows for focusing on particular 
                                        tracks of interest. If None, the operation applies to tracks related to the specified 
                                        agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the analysis. Defaults to 1.
            out_path (str, optional): The file path where the dataset with the extracted characteristic tracks should be saved. 
                                        If None, it assumes the current data_path.
            inplace (bool, optional): Whether to modify the dataset in place, or add a Characteristic column.
                                      Defaults to False, indicating not modifying in place.
            desc (str, optional): A brief description of the operation. Defaults to 'Extracting characteristic tracks'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #process in parallel
        utils.pool_caller(geometry.characteristic_tracks,
                          (stop_threshold,
                           turn_threshold,
                           min_distance,
                           max_distance,
                           min_stop_duration,
                           out_path,
                           inplace),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        meta = self.meta.copy()
        meta['Characteristic'] = 'Characteristic Points'
        self = self._update_meta(out_path, meta)
        return self
    
    def simplify_stops(self, 
                       stop_threshold=0.15, 
                       min_stop_duration=1800,
                       max_drift_distance=1000,
                       agents=None,
                       tracks=None, 
                       ncores=1, 
                       out_path=None,
                       desc='Simplifying stops along tracks'):
        """
        Simplifies and reduces the number of points that define stop events along tracks.

        This can be useful for simplifying large amounts of unnecessary data into key
        points such as start/end points, which maintain the critical information of the stop 
        (e.g. duration). In particular, this has proved useful to simplify AIS vessel tracks
        where AIS transponders have been left on at a mooring location for an extended period of time.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.
        
        Here all spatial/speed kwargs are in the CRS units. I.e. degrees for geographic, meters for UTM, etc.
        
        Args:
            stop_threshold (float, optional): The speed threshold below which a movement is considered a stop. Defaults to 0.15.
            min_stop_duration (int, optional): The minimum duration (in seconds) of a stop to be considered significant. 
                                                            Defaults to 1800 seconds (30 minutes).
            max_drift_distance (int, optional): The maximum allowable distance between stop points before an intermediate point
                                                is maintained. Defaults to 1000.
            agents (list, optional): A list of agent IDs for which stops will be simplified. If None, the simplification process 
                                    is applied to stops associated with all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which stops will be simplified. This allows for selective 
                                    simplification of stops within certain tracks. If None, the operation applies to tracks related 
                                    to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the simplification operation. Defaults to 1, 
                                    implying serial processing. Utilizing more cores can speed up the process for large datasets.
            out_path (str, optional): The file path where the dataset with simplified stops should be saved. If None, it assumes
                                    the current data_path.
            desc (str, optional): A brief description of the operation. Defaults to 'Simplifying stops along tracks'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.simplify_stops,
                          (stop_threshold,
                           min_stop_duration,
                           max_drift_distance,
                           out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def imprint_geometry(self, 
                         shape,
                         agents=None,
                         tracks=None, 
                         ncores=1, 
                         out_path=None,
                         desc='Imprinting geometry into tracks'):
        """
        Imprints a specified geometric shape onto track data. Shape must be a shapely LineString, 
        MultiLineString, Polygon, or MultiPolygon. The shape must be in the same CRS as the data.

        This is very useful for things like clipping tracks, isolating parts of tracks inside of polygons, 
        or getting an accurate estimate of time spent inside a polygon. This may also be useful if you simply
        want to add data to tracks at a very specific location for later use.

        Here, linear interpolation is used to fill any dynamic data at new points.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            shape: The geometric shape to be imprinted onto the tracks. Shape must be a shapely LineString, 
                  MultiLineString, Polygon, or MultiPolygon.
            agents (list, optional): A list of agent IDs whose tracks will be modified by the geometric shape.
                                    If None, the operation applies to all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be modified by the geometric shape. This
                                    allows for selective application of the geometry to certain tracks. If None,
                                    the operation applies to tracks related to the specified agents or all tracks
                                    if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the operation. Defaults to 1.
            out_path (str, optional): The file path where the dataset with the imprinted geometric information
                                    should be saved. If None, it assumes the current data_path.
            desc (str, optional): A brief description of the operation. Defaults to 'Imprinting geometry into tracks'.

        Returns:
            self: The Dataset instance.
        """
        #check input is valid
        msg = 'geometry must be a shapely LineString, MultiLineString, Polygon, or MultiPolygon'
        assert isinstance(shape, (LineString, 
                                  MultiLineString, 
                                  Polygon, 
                                  MultiPolygon)), msg
        polylines = utils.prepare_polylines(shape)
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.imprint_geometry,
                          (polylines, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self

    def interpolate_raster(self,
                           ras,
                           name,
                           agents=None,
                           tracks=None,
                           ncores=1,
                           out_path=None,
                           method='linear',
                           meta='Raster Values',
                           desc='Interpolating raster to tracks'):
        """
        Interpolates values from a raster onto the track points, effectively assigning raster-based
        values (e.g., elevation, temperature) to each point along the tracks. This method allows for the enrichment
        of track data with additional environmental or spatial information.

        Here, there is no temporal aspect to the interpolation. The raster is simply interpolated to all
        points on the tracks.

        This is a simple wrapper over scipy.interpolate.RegularGridInterpolator. The method kwarg corresponds to the
        options available for the scipy version:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        Args:
            ras: The raster dataset from which values will be interpolated. This must be a rasterio object.
            name (str): The name to be assigned to the interpolated raster values in the track data.
            agents (list, optional): A list of agent IDs for which the raster values will be interpolated. If None,
                                    interpolation is applied across all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which the raster values will be interpolated. This
                                    allows for selective application of the raster data to certain tracks. If None, the
                                    operation applies to tracks related to the specified agents or all tracks if no agents
                                    are specified.
            ncores (int, optional): The number of processing cores to use for the interpolation operation. Defaults to 1.
            out_path (str, optional): The file path where the dataset with interpolated raster values should be saved. If None,
                                    the current data_path is assumed.
            method (str, optional): The method of interpolation to perform. Supported are “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”.
                                    Defaults to 'linear'.
            meta (str, optional): A description of this data field for the Dataset.meta attribute. E.g. temperature, bathymetry, wind speed, etc.
            desc (str, optional): A brief description of the operation. Defaults to 'Interpolating raster to tracks'.

        Returns:
            self: The Dataset instance.
        """
        #assert rasterio object
        msg = 'raster must be a rasterio object'
        assert isinstance(ras, rio.io.DatasetReader), msg
        #assert same crs
        msg = 'raster must have same CRS as Dataset'
        assert pyproj.CRS(self.meta['CRS']).equals(ras.crs), msg
        x,_ = ras.xy([0]*ras.shape[0],range(ras.shape[0]))
        _,y = ras.xy(range(ras.shape[1]), [0]*ras.shape[1])
        interp = RegularGridInterpolator((x,y), 
                                         ras.read(1),
                                         bounds_error=False,
                                         fill_value=np.nan,
                                         method=method)
        #set out path
        out_path = self._set_out_path(out_path)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #resample in parallel
        utils.pool_caller(geometry.interpolate_raster,
                          (interp, name, out_path),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        _meta = self.meta.copy()
        _meta[name] = meta
        self = self._update_meta(out_path, _meta)
        return self

    def encounters(self, #first ds
                   dataset=None, #second ds
                   distance=0.1, #in dataset CRS units
                   time=3600, #seconds
                   ncores=1, 
                   tracks0=None, #tracks in first ds
                   tracks1=None, #tracks in second ds
                   data_cols=[],
                   meta_cols=[],
                   filter_min=False, #only keep closest encounter
                   desc='Calculating spatiotemporal encounters'):
        """
        Calculates spatiotemporal encounters between two datasets. Identifies instances where tracks 
        come within a specified distance of each other within certain difference in time. This function can be 
        used for analyzing interactions or proximities between tracks.

        By default, dataset=None is passed, which forces the second dataset to be the same as the first. Meaning,
        the default is to find encounters by comparing a dataset to itself; this is likely the most common application.

        You can optionally pass filter_min=True which only maintains the closest encounter (closest point of approach or CPA)
        between tracks. If False, it will return all encounters within the specified distance and time thresholds.

        By passing a list of meta_cols or data_cols, you can also maintain metadata about the interactions (e.g. speed, coursing)
        which can facilitate more complex analyses with the data.

        Here all spatial kwargs are in the CRS units. I.e. degrees for geographic, meters for UTM, etc.

        This method is relatively slow, so it's recommended to carefully choose which tracks to process.

        Args:
            dataset: The trackio.Dataset to compare against. By default, None is passed, which makes the algorithm compare
                    the input Dataset against itself.
            distance (float, optional): The maximum spatial distance between two tracks for an encounter to be considered valid.
                                         Defaults to 0.1.
            time (int, optional): The maximum temporal distance (in seconds) between encounters for them to be considered valid.
                                 Defaults to 1 hour = 3600 seconds.
            ncores (int, optional): The number of processing cores to use for calculating encounters. Defaults to 1.
            tracks0 (list, optional): A list of track IDs from the input dataset to be considered. If None, all tracks in the dataset are considered.
            tracks1 (list, optional): A list of track IDs from the comparing dataset to be considered. If None, all tracks in the dataset are considered.
            data_cols (list of str, optional): Specific data columns from the dataset to include in the encounter output.
            meta_cols (list of str, optional): Specific metadata columns to include in the encounter output.
            filter_min (bool, optional): If True, filters the encounter results to the closest encounters between pairs of agents.
                                        If False, it will return all encounters below the distance and time thresholds. Defaults to False.
            desc (str, optional): A brief description of the operation. Defaults to 'Calculating spatiotemporal encounters'.

        Returns:
            pd.DataFrame: Pandas DataFrame containing spatiotemporal encounters, along with requested meta_cols and data_cols.
        """
        #if no input, self encounters
        if dataset is None:
            dataset = self.copy()
        assert self.meta['CRS'] == dataset.meta['CRS'], 'CRS does not match between Datasets'
        #get the two track databases
        db1 = self.tracks
        db2 = dataset.tracks
        #reduce databases
        keepcols = ['Start Time',
                    'End Time', 
                    'File',
                    'Agent ID']
        db1 = db1[keepcols]
        db2 = db2[keepcols]
        if tracks0 is not None:
            db1 = db1.loc[tracks0]
        if tracks1 is not None:
            db2 = db2.loc[tracks1]   
        #find overlapping unique pairs
        _pairs = {'Track ID_0':[],
                  'Track ID_1':[],
                  'Agent ID_0':[],
                  'Agent ID_1':[],
                  'File_0':[],
                  'File_1':[]}
        for i in range(len(db1)):
            row = db1.iloc[i]
            tid1 = row.name
            aid1 = row['Agent ID']
            f1 = row['File']
            start = row['Start Time']
            end = row['End Time'] 
            time_ids = np.nonzero((end >= db2['Start Time'] - pd.Timedelta(f'{time}s')).values 
                                   & (db2['End Time'] + pd.Timedelta(f'{time}s') >= start).values)[0]
            tid2 = db2.index[time_ids]
            aid2 = db2['Agent ID'].iloc[time_ids]
            f2 = db2['File'].iloc[time_ids]
            tid1 = [tid1]*len(tid2)
            aid1 = [aid1]*len(aid2)
            f1 = [f1]*len(f2)
            _pairs['Track ID_0'].extend(tid1)
            _pairs['Track ID_1'].extend(tid2)
            _pairs['Agent ID_0'].extend(aid1)
            _pairs['Agent ID_1'].extend(aid2)
            _pairs['File_0'].extend(f1)
            _pairs['File_1'].extend(f2)
        pairs = pd.DataFrame(_pairs)
        #make a unique key for all pairs
        pairs['sorted'] = pairs.apply(lambda x: '_'.join(sorted(x)), axis=1)
        #delete duplicated keys
        pairs = pairs.drop_duplicates(subset='sorted')
        #delete self encounters
        pairs = pairs[pairs['Track ID_0'] != pairs['Track ID_1']]
        #simplify track ids for function
        pairs['ID_0'] = pairs['Track ID_0'].apply(lambda x: x.rsplit('_')[-1])
        pairs['ID_1'] = pairs['Track ID_1'].apply(lambda x: x.rsplit('_')[-1])
        #group by agent id to reduce file opens
        grouped = pairs.groupby(['Agent ID_0', 'Agent ID_1']).agg(list)
        grouped['File_0'] = grouped['File_0'].apply(lambda x: x[0])
        grouped['File_1'] = grouped['File_1'].apply(lambda x: x[0])
        #process in parallel
        rows = utils.pool_caller(geometry.encounters,
                                 (grouped, 
                                  distance, 
                                  time, 
                                  data_cols,
                                  meta_cols,
                                  filter_min),
                                 list(range(len(grouped))),
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
                      data_cols=[],
                      meta_cols=[],
                      desc='Calculating intersections'):
        """
        Calculates intersections between two datasets. Identifies instances where tracks intersect with other tracks,
        within certain difference in time. This function can be  used for analyzing interactions or proximities between tracks.

        By default, dataset=None is passed, which forces the second dataset to be the same as the first. Meaning,
        the default is to find intersections by comparing a dataset to itself; this is likely the most common application.

        This is essentially the same as the Dataset.encounters method, but here the tracks must actually cross/intersect.

        By passing a list of meta_cols or data_cols, you can also maintain metadata about the interactions (e.g. speed, coursing)
        which can facilitate more complex analyses with the data.

        Here all spatial kwargs are in the CRS units. I.e. degrees for geographic, meters for UTM, etc.

        This method is relatively slow, so it's recommended to carefully choose which tracks to process.

        Args:
            dataset: The trackio.Dataset to compare against. By default, None is passed, which makes the algorithm compare
                    the input Dataset against itself.
            time (int, optional): The maximum temporal distance (in seconds) between intersections for them to be considered valid.
                                 Defaults to 1 hour = 3600 seconds.
            ncores (int, optional): The number of processing cores to use. Defaults to 1.
            tracks0 (list, optional): A list of track IDs from the input dataset to be considered. If None, all tracks in the dataset are considered.
            tracks1 (list, optional): A list of track IDs from the comparing dataset to be considered. If None, all tracks in the dataset are considered.
            data_cols (list of str, optional): Specific data columns from the dataset to include in the output.
            meta_cols (list of str, optional): Specific metadata columns to include in the output.
            desc (str, optional): A brief description of the operation. Defaults to 'Calculating intersections'.

        Returns:
            pd.DataFrame: Pandas DataFrame containing intersections, along with requested meta_cols and data_cols.
        """
        #if no input, self intersection
        if dataset is None:
            dataset = self.copy()
        assert self.meta['CRS'] == dataset.meta['CRS'], 'CRS does not match between DataSets'
        #get the two track databases
        db1 = self.tracks
        db2 = dataset.tracks
        #reduce databases
        keepcols = ['Start Time',
                    'End Time', 
                    'File',
                    'Agent ID',
                    'geometry']
        db1 = db1[keepcols]
        db2 = db2[keepcols]
        if tracks0 is not None:
            db1 = db1.loc[tracks0]
        if tracks1 is not None:
            db2 = db2.loc[tracks1]   
        #find overlapping unique pairs
        _pairs = {'Track ID_0':[],
                  'Track ID_1':[],
                  'Agent ID_0':[],
                  'Agent ID_1':[],
                  'File_0':[],
                  'File_1':[]}
        for i in range(len(db1)):
            row = db1.iloc[i]
            tid1 = row.name
            aid1 = row['Agent ID']
            f1 = row['File']
            start = row['Start Time']
            end = row['End Time'] 
            bbox = row['geometry']
            time_ids = np.nonzero((end >= db2['Start Time'] - pd.Timedelta(f'{time}s')).values 
                                   & (db2['End Time'] + pd.Timedelta(f'{time}s') >= start).values)[0]
            space_ids = np.nonzero(db2.geometry.intersects(bbox).values)[0]
            all_ids = [id for id in space_ids if id in time_ids]
            tid2 = db2.index[all_ids]
            aid2 = db2['Agent ID'].iloc[all_ids]
            f2 = db2['File'].iloc[all_ids]
            tid1 = [tid1]*len(tid2)
            aid1 = [aid1]*len(aid2)
            f1 = [f1]*len(f2)
            _pairs['Track ID_0'].extend(tid1)
            _pairs['Track ID_1'].extend(tid2)
            _pairs['Agent ID_0'].extend(aid1)
            _pairs['Agent ID_1'].extend(aid2)
            _pairs['File_0'].extend(f1)
            _pairs['File_1'].extend(f2)
        pairs = pd.DataFrame(_pairs)
        #make a unique key for all pairs
        pairs['sorted'] = pairs.apply(lambda x: '_'.join(sorted(x)), axis=1)
        #delete duplicated keys
        pairs = pairs.drop_duplicates(subset='sorted')
        #delete self encounters
        pairs = pairs[pairs['Track ID_0'] != pairs['Track ID_1']]
        #simplify track ids for function
        pairs['ID_0'] = pairs['Track ID_0'].apply(lambda x: x.rsplit('_')[-1])
        pairs['ID_1'] = pairs['Track ID_1'].apply(lambda x: x.rsplit('_')[-1])
        #group by agent id to reduce file opens
        grouped = pairs.groupby(['Agent ID_0', 'Agent ID_1']).agg(list)
        grouped['File_0'] = grouped['File_0'].apply(lambda x: x[0])
        grouped['File_1'] = grouped['File_1'].apply(lambda x: x[0])
        #process in parallel
        rows = utils.pool_caller(geometry.intersections,
                                 (grouped, 
                                  time, 
                                  data_cols,
                                  meta_cols),
                                 list(range(len(grouped))),
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
                            data_cols=[],
                            meta_cols=[],
                            ncores=1, 
                            desc='Calculating proximities to object'):
        """
        Calculates the minimum proximity of tracks to a specified geometric object. This object can be a
        shapely Point, MultiPoint, LineString, MultiLineString, Polygon, or MultiPolygon. It can also be
        a Nx2 numpy array containing x,y points that represent the former.
         
        The minimum proxixity between tracks and the object may not necessarily be points that fall on either of 
        the original shapes. For example, if the object is a point, the closest proximity may be somewhere in the 
        middle of one of the track's segments. This is similar to the shapely.ops.nearest_points:

        https://shapely.readthedocs.io/en/stable/manual.html#shapely.ops.nearest_points

        Here, linear interpolation is used to fill any dynamic data at new points if they don't fall on the original shapes.

        Args:
            shapely_object: A geometric object defining the feature to which proximities are calculated. See description above for options.
            agents (list, optional): A list of agent IDs for which proximity calculations will be performed. If None,
                                    the function will calculate proximities for all agents in the dataset.
            tracks (list, optional): A list of specific track IDs for which proximities will be calculated. This allows
                                    for selective analysis of certain tracks. If None, the operation applies to tracks
                                    related to the specified agents or all tracks if no agents are specified.
            data_cols (list, optional): A list of data column names to include in the output.
            meta_cols (list, optional): A list of metadata column names to include in the output.
            ncores (int, optional): The number of processing cores to use for the calculation. Defaults to 1.
            desc (str, optional): A brief description of the operation. Defaults to 'Calculating proximities to object'.

        Returns:
            DataFrame: A pandas DataFrame containing the minimum proximities, along with the requested meta_cols and data_cols.
        """
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
                    data_cols=[],
                    meta_cols=[],
                    ncores=1, 
                    bins=None, 
                    relative=False,
                    desc='Calculating track proximities'):
        """
        ***Note, this method assumes Datset.resample_time_global has already been run. If the tracks aren't on the same
        time axis, this will not work***

        Calculates proximities between tracks. This analysis can help identify close encounters, common pathways, 
        or general spatiotemporal relationships between moving entities represented by the tracks. 

        By default, dataset=None is passed, which forces the second dataset to be the same as the first. Meaning,
        the default is to find the proximity of tracks inside the same Dataset; this is likely the most common application.

        By passing a list of meta_cols or data_cols, you can also maintain metadata about the interactions (e.g. speed, coursing)
        which can facilitate more complex analyses with the data.

        If bins is left as None, the output will be the closest point of approach (CPA) between each pair of tracks. 

        If bins is not None, it must be a list of distance bins. Then, the output will be the amount of time spent
        in each distance between, for each pair of tracks.

        If bins is not None and relative=True, then the output will be the same as above, however it will be reduced
        to the amount of time spent between ANY pair of tracks in the dataset, giving a global measure of proximity across all tracks
        in the dataset.

        This is a very useful method for measuring the amount of time agents spend at different distances from one another.

        This method is relatively slow, so it's recommended to carefully choose which tracks to process.

        Args:
            dataset: The trackio.Dataset to compare against. By default, None is passed, which makes the algorithm compare
                    the input Dataset against itself.
            tracks0 (list, optional): A list of track IDs from the input dataset to be considered. If None, all tracks in the dataset are considered.
            tracks1 (list, optional): A list of track IDs from the comparing dataset to be considered. If None, all tracks in the dataset are considered.
            data_cols (list of str, optional): Specific data columns from the dataset to include in the output.
            meta_cols (list of str, optional): Specific metadata columns to include in the output.
            ncores (int, optional): The number of processing cores to use for the calculation. Defaults to 1.
            bins (list of int/float, optional): An optional list of distance bins for categorizing distances. If specified,
                                proximities will be aggregated into these bins, facilitating the analysis of
                                distance distributions. Defaults to None, which outputs only the CPA between pairs of tracks.
            relative (bool, optional): Only used if bins is not None. If False, returns the proximities between all pairs of tracks.
                                      If True, reduces output to proximities across all tracks in the dataset.
            desc (str, optional): A brief description of the operation. Defaults to 'Calculating track proximities'.

        Returns:
            DataFrame: A pandas DataFrame containing the proximity output, depending on the value of bins and relative kwargs.
        """
        print('Proximity analysis assumes that self.resample_time_global has already been run, '\
            'if not the results will be invalid or the function may fail')
        #if no input, self encounters
        if dataset is None:
            dataset = self.copy()
        assert self.meta['CRS'] == dataset.meta['CRS'], 'CRS does not match between Datasets'
        #ensure relative is False if no bins
        if bins is None:
            relative = False
        #get the two track databases
        db1 = self.tracks
        db2 = dataset.tracks
        #reduce databases
        keepcols = ['Start Time',
                    'End Time', 
                    'File',
                    'Agent ID']
        db1 = db1[keepcols]
        db2 = db2[keepcols]
        if tracks0 is not None:
            db1 = db1.loc[tracks0]
        if tracks1 is not None:
            db2 = db2.loc[tracks1]   
        #find overlapping unique pairs
        _pairs = {'Track ID_0':[],
                  'Track ID_1':[],
                  'Agent ID_0':[],
                  'Agent ID_1':[],
                  'File_0':[],
                  'File_1':[]}
        for i in range(len(db1)):
            row = db1.iloc[i]
            tid1 = row.name
            aid1 = row['Agent ID']
            f1 = row['File']
            start = row['Start Time']
            end = row['End Time'] 
            time_ids = np.nonzero((end >= db2['Start Time']).values 
                                   & (db2['End Time'] >= start).values)[0]
            tid2 = db2.index[time_ids]
            aid2 = db2['Agent ID'].iloc[time_ids]
            f2 = db2['File'].iloc[time_ids]
            tid1 = [tid1]*len(tid2)
            aid1 = [aid1]*len(aid2)
            f1 = [f1]*len(f2)
            _pairs['Track ID_0'].extend(tid1)
            _pairs['Track ID_1'].extend(tid2)
            _pairs['Agent ID_0'].extend(aid1)
            _pairs['Agent ID_1'].extend(aid2)
            _pairs['File_0'].extend(f1)
            _pairs['File_1'].extend(f2)
        pairs = pd.DataFrame(_pairs)
        #make a unique key for all pairs
        pairs['sorted'] = pairs.apply(lambda x: '_'.join(sorted(x)), axis=1)
        #delete duplicated keys
        pairs = pairs.drop_duplicates(subset='sorted')
        #delete self encounters
        pairs = pairs[pairs['Track ID_0'] != pairs['Track ID_1']]
        #simplify track ids for function
        pairs['ID_0'] = pairs['Track ID_0'].apply(lambda x: x.rsplit('_')[-1])
        pairs['ID_1'] = pairs['Track ID_1'].apply(lambda x: x.rsplit('_')[-1])
        #group by agent id to reduce file opens
        grouped = pairs.groupby(['Agent ID_0', 'Agent ID_1']).agg(list)
        grouped['File_0'] = grouped['File_0'].apply(lambda x: x[0])
        grouped['File_1'] = grouped['File_1'].apply(lambda x: x[0])
        #process in parallel
        rows = utils.pool_caller(geometry.proximities,
                                 (grouped,
                                  bins,
                                  relative,
                                  data_cols,
                                  meta_cols),
                                 list(range(len(grouped))),
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
        """
        Analyzes the lateral distribution of tracks across specified slices/cross-sections of the data.
         his method can also split tracks going to/from the direction along start to end to separate 
        opposing "lanes" of traffic.
        
        Slices are taken along a line from start to end, with either n_slices or spacing defining where the slices fall. 
        You can specify the lateral distribution bin width, and whether to return a relative (0-1) or absolute probability
        distribution function.

        You can also provide a list of meta_cols and/or data_cols to collect information about tracks at these slices.

        You can, and in the author's opinion should, pass a polygon to clip the slices to. This is because the slices
        extend to infinity, and with complex tracks this could result in intersections in odd places. The polygon
        can be a shapely Polygon, or Nx2 numpy defining a polygon.

        Args:
            start: The starting point (x,y).
            end: The ending point (x,y).
            split (bool, optional): If True, a Direction column is added to the output indicating to (T) or from (F) direction
            spacing (float, optional): The spacing between the slices. Defaults to 100.
            n_slices (int, optional): The number of slices to generate. If this is > 0, it will override spacing.
            agents (list, optional): A list of agent IDs to be included in the analysis. If None, the analysis
                                    includes all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be analyzed. If None, the analysis includes
                                    tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the calculation. Defaults to 1.
            polygon: (optional): A polygon used to clip the extents of the slices. Defaults to None, meaning slices
                                extend to infinity. The polygon can be a shapely Polygon, or Nx2 numpy defining a polygon.
            density (bool, optional): If True, the lateral distribution is normalize to sum to 1. If False, it returns absolute counts.
            bins (float, optional): The bin spacing along each slice for the lateral distribution.
            meta_cols (list, optional): A list of metadata column names to include in the output.
            data_cols (list, optional): A list of data column names to include in the output.
            desc (str, optional): A brief description of the operation. Defaults to 'Calculating lateral distributions
                                at slices'.

        Returns:
            DataFrame: A pandas DataFrame containing the distribution data across slices, along with the requested meta_cols and data_cols.
        """
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
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Computes the time spent and distance travelled by tracks inside a specified polygonal area. This function is useful
        for analyzing spatial usage, such as habitat utilization, restricted area compliance, or general movement patterns
        within specific geographic boundaries. The analysis can help identify which agents or tracks spend time in the
        area of interest and quantify that time.

        This is best used in conjunction with Dataset.imprint_geometry, as it can imprint the polygon outline into 
        the actual tracks, thereby making the time spent and distance travelled estimates more accurate.

        The polygon can be a shapely Polygon, or Nx2 numpy defining a polygon.

        Optionally, you can provide a list of meta_cols and data_cols to return with the output. If you pass a list of
        data_cols, it will return the first value on/inside the polygon, and the last on/inside the polygon for each
        track that passes through.

        Args:
            polygon: (optional): A polygon used to clip the extents of the slices. Defaults to None, meaning slices
                                extend to infinity. The polygon can be a shapely Polygon, or Nx2 numpy defining a polygon.
            agents (list, optional): A list of agent IDs to be included in the time calculation. If None, the analysis
                                    will consider all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be analyzed. If None, the analysis includes
                                    tracks related to the specified agents or all tracks if no agents are specified.
            meta_cols (list, optional): A list of metadata column names to include in the output.
            data_cols (list, optional): A list of data column names to include in the output.
            desc (str, optional): A brief description of the operation. Defaults to 'Computing time spent in polygon'.
            ncores (int, optional): The number of processing cores to use for the calculation. Defaults to 1.

        Returns:
            DataFrame: A pandas DataFrame containing the calculated times spent and distances travelled within the polygon
                    for each track, along with any specified meta_cols and data_cols.
        """
        #get polygon
        polygon, edges = utils.format_polygon(polygon)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Generates a flow map based on movement or connectivity between specified polygonal areas. This function is
        useful for visualizing movement patterns, such as migration routes, traffic flows, or general movements of
        agents through different regions. The flow map can highlight areas of high connectivity or movement concentration,
        providing insights into spatial dynamics within the dataset from a macroscopic perspective.

        This method assumes that you have already ran the Dataset.classify_in_polygons method to classify the 
        tracks inside the polygons GeoDataFrame. The flow_col kwarg is the column that was produced when running 
        Dataset.classify_in_polygons.

        This method works by collecting all of the movements from polygon to polygon in the track data. Then, the movements
        are grouped by their unique transitions (e.g. A-B, B-A, A-C, C-A, B-C, B-C, etc.). The result is a geopandas
        GeoDataFrame of LineStrings representing the flow map edges, and the number of movements (volume) along
        each edge.

        Optionally, you can pass a boolean characteristic_col which will make the algorithm only consider points where
        this column is True in the track data. This is usually used after running Dataset.characteristic_tracks.

        Args:
            polygons: polygons must be geopandas GeoDataFrame with [Code, X, Y columns].
            characteristic_col (str, optional): The boolean column name representing the characteristic points in a given track.
                                                Typically "Characteristic". Defaults to None, making the algorithm consider all
                                                points in the tracks.
            flow_col (str, optional): The column name containing the integer polygon codes.
            agents (list, optional): A list of agent IDs to include in the flow analysis. If None, the analysis
                                    considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be analyzed for generating the flow map. If None,
                                    the analysis includes tracks related to the specified agents or all tracks if no
                                    agents are specified.
            ncores (int, optional): The number of processing cores to use for the analysis. Defaults to 1.
            desc (str, optional): A brief description of the operation. Defaults to 'Generating flow map from polygons'.

        Returns:
            GeoDataFrame: A geopandas GeoDataFrame containing LineStrings representing the unique edges of the flow map,
                         along with their volume.
        """
        #make sure it is proper format
        msg = 'Polygons must be gp.GeoDataFrame with Code, X, Y columns'
        assert isinstance(polygons, gp.GeoDataFrame), msg
        assert all([c in polygons.columns for c in ['Code','X','Y']]), msg
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
                          characteristic_col=None,
                          flow_col='Graph Node',
                          agents=None,
                          tracks=None,
                          ncores=1,
                          out_path=None,
                          desc='Generating flow map from polygons'):
        """
        This works the same way as Dataset.generate_flow_map, but instead the tracks are actually
        reduced to their flow map representation.

        This method assumes that you have already ran the Dataset.classify_in_polygons method to classify the 
        tracks inside the polygons GeoDataFrame. The flow_col kwarg is the column that was produced when running 
        Dataset.classify_in_polygons.

        This method works by routing each track through the polygons using the flow_col data, and then reducing
        the track to its flow map representation by using the X and Y columns in the polygons input argument.

        Optionally, you can pass a boolean characteristic_col which will make the algorithm only consider points where
        this column is True in the track data. This is usually used after running Dataset.characteristic_tracks.

        Args:
            polygons: polygons must be geopandas GeoDataFrame with [Code, X, Y columns].
            characteristic_col (str, optional): The boolean column name representing the characteristic points in a given track.
                                                Typically "Characteristic". Defaults to None, making the algorithm consider all
                                                points in the tracks.
            flow_col (str, optional): The column name containing the integer polygon codes.
            agents (list, optional): A list of agent IDs to include in the flow analysis. If None, the analysis
                                    considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be analyzed for generating the flow map. If None,
                                    the analysis includes tracks related to the specified agents or all tracks if no
                                    agents are specified.
            ncores (int, optional): The number of processing cores to use for the analysis. Defaults to 1.
            desc (str, optional): A brief description of the operation. Defaults to 'Generating flow map from polygons'.

        Returns:
            self: The Dataset instance.
        """
        #set out path
        out_path = self._set_out_path(out_path)
        #make sure it is proper format
        msg = 'Polygons must be gp.GeoDataFrame with Code, X, Y columns'
        assert isinstance(polygons, gp.GeoDataFrame), msg
        assert all([c in polygons.columns for c in ['Code','X','Y']]), msg
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #make point lookup
        keys = polygons['Code'].values
        vals = polygons[['X','Y']].values
        points = dict(zip(keys, vals))
        #compute in parallel
        utils.pool_caller(geometry.reduce_to_flow_map,
                        (characteristic_col, 
                        flow_col,
                        out_path,
                        points),
                        pkl_groups,
                        desc,
                        ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self
    
    def route_through_raster(self, 
                             ras,
                             agents=None,
                             tracks=None,
                             out_path=None,
                             end=None,
                             desc='Routing tracks through raster',
                             ncores=1,
                             **kwargs):
        """
        Routes tracks through a cost raster using a least cost path algorithm. This is a thin wrapper
        over skimage.graph.route_through_array, and will accept any of the kwargs the original function
        will accept:

        https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.route_through_array 

        This can be used to generate synthetic movements through obstacle arrays, or to very simply
        model the movement of agents through some form of force/cost/resistance field.

        The method only applies to tracks that intersect with the outline of the raster.

        The simplest case is when a track passes through the raster completely. In this case, the 
        track is routed from the entrance to the exit of the raster outline.
        
        The first edge case is when a track starts inside the raster and leaves. In this case, the
        track is routed from the first point on the track to the exit of the raster outline.

        The last edge case is when a track starts and ends inside the raster outline. By default, the
        algorithm will route the track from it's first to last point through the raster. But, you
        can optionally override the end point by passing an (x,y) pair with the end kwarg. This is
        useful, for example, if AIS data cuts off near a known port but you want to "complete" the journey
        with some synthetic data.

        Temporal linear interpolation is applied to tracks between start/end points in the raster. In general,
        the timing and speeds derived from points coming from this algorithm should not be trusted. The
        algorithm is more geared towards geometric/spatial rerouting.

        If you pass a directory for the out_path kwarg, the *.tracks files will be saved to this directory,
        and self.data_path will be changed as well. If you pass None, it simply saves the *.tracks files 
        in the original self.data_path location.

        Args:
            ras: The raster dataset that tracks will be routed through. This must be a rasterio object.
            agents (list, optional): A list of agent IDs whose tracks will be considered for routing. If None,
                                    the routing process is applied to tracks associated with all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be routed through the raster. This allows for
                                    selective application of the routing process to certain tracks. If None, the
                                    operation applies to tracks related to the specified agents or all tracks if no
                                    agents are specified.
            out_path (str, optional): The file path where the dataset with the routed tracks should be saved. If None,
                                    it assumes the current data_path.
            end (optional): An optional (x,y) end location that will override the default end location for the routing.
            desc (str, optional): A brief description of the operation. Defaults to 'Routing tracks through raster'.
            ncores (int, optional): The number of processing cores to use for the routing operation. Defaults to 1.
            **kwargs: Additional keyword arguments accepted by skimage.graph.route_through_array.

        Returns:
            self: The Dataset instance.
        """
        #assert rasterio object
        msg = 'raster must be a rasterio object'
        assert isinstance(ras, rio.io.DatasetReader), msg
        #set out path
        out_path = self._set_out_path(out_path)
        #get the raster info
        b = ras.bounds
        polygon, edges = utils.format_polygon(box(b.left, b.bottom, b.right, b.top))
        array = ras.read(1)
        transform = ras.transform
        width = ras.width
        height = ras.height
        row_inds, col_inds = np.indices((height, width))
        x_coords, y_coords = transform * (col_inds, row_inds)
        coords = np.dstack([x_coords, y_coords])
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
        #compute in parallel
        utils.pool_caller(geometry.route_through_raster,
                          (array, 
                           polygon, 
                           edges, 
                           coords,
                           end,
                           out_path,
                           kwargs),
                          pkl_groups,
                          desc,
                          ncores)
        #update meta
        self = self._update_meta(out_path, self.meta)
        return self

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
        """
        Classifies points along tracks True if they fall on/within a specified polygon,
        and False if they do not. This classification is stored in a boolean column in the track data.

        The polygon must be a shapely Polygon, or a Nx2 array representing the outline of a polygon.

        Args:
            polygon: The polygon must be a shapely Polygon, or a Nx2 array representing the outline of a polygon.
            agents (list, optional): A list of agent IDs to be included in the classification. If None, the classification
                                    will consider all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified. This allows for selective analysis
                                    of certain tracks. If None, the operation applies to tracks related to the specified
                                    agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=16, then 
                                 the output column is Code16. Defaults to 16. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks inside polygon'.

        Returns:
            self: The Dataset instance.
        """
        #get polygon
        polygon, edges = utils.format_polygon(polygon)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
                            to_codes=False,
                            name='Polygon',
                            desc='Classifying tracks inside polygons'):
        """
        Classifies points if they fall om/within user defined polygons.

        Here, polygons is a geopandas GeoDataFrame with a series of shapely Polygons for geometry,
        and a Code column with unique integers for each polygon. The CRs must be the same
        as the dataset CRS.

        By default, a column is created using the name kwarg which contains the Code values corresponding to
        which polygon each point is in. If a point falls inside none of the polygons, then the value will be 0.
        This also means that 0 should not be used as a Code value for any polygon to avoid conflicts.

        Optionally, you can pass to_codes=True. This will create a Code column for every polygon in polygons
        with CodeN as the name, where N is the Code for each polygon. These are boolean columns and will
        be True when the points are inside that polygon, and False if not. This gives you the same
        behaviour as Dataset.classify_in_polygon, just with more than 1 polygon.

        Args:
            polygons: geopandas GeoDataFrame with a series of shapely Polygons for geometry, and a Code column with unique 
                     integers for each polygon. Do not use 0 as a Code value.
            agents (list, optional): A list of agent IDs to be included in the classification. If None, the classification
                                    will consider all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified. This allows for selective analysis
                                    of certain tracks. If None, the operation applies to tracks related to the specified
                                    agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults to 1.
            to_codes (bool, optional): If False, only a column is created using the name kwarg. If True, a boolean Code column
                                      is created for each polygon. Defaults to False.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks inside polygons'.
            name (str): Name of the column to be created in the track data. This column will hold polygon code values depending
                       on which polygon the points land in. If it lands in no polygons, the value of this column will be 0 at
                       those points.

        Returns:
            self: The Dataset instance.
        """
        #copy
        polygons = polygons.copy()
        #get polygons
        polygons['polys'] = polygons.geometry.apply(utils.format_polygon)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Classifies points on tracks based on a speed threshold, marking them according to whether their speeds are
        higher or lower than the specified threshold. This function is useful for identifying high-speed movements,
        slow-paced activities, or generally categorizing tracks by their speed characteristics for further analysis.

        This classification is stored in a boolean Code column.

        Here, the speed threshold is in the same units as the Dataset CRS. I.e. degree/s for geographic,
        meter/s for UTM, etc.

        Args:
            speed (float): The speed threshold used for classification.
            higher (bool, optional): If True, tracks or agents with speeds higher than the specified threshold will
                                    be classified. Defaults to True.
            lower (bool, optional): If True, tracks or agents with speeds lower than the specified threshold will
                                    be classified. Defaults to False.
            agents (list, optional): A list of agent IDs to be included in the speed classification. If None,
                                    the classification considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified by speed. This allows for selective
                                    analysis of certain tracks. If None, the operation applies to tracks related to the
                                    specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults
                                    to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=17, then 
                                 the output column is Code17. Defaults to 17. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks by speed threshold',
                                which can be customized for more specific descriptions or contexts of the classification.

        Returns:
            self: The Dataset instance.
        """
        #check for bounds
        if lower:
            higher = False
        elif higher:
            lower = False
        else:
            raise Exception('Must pass higher=True or lower=True, but not both')
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Classifies points on tracks based on their turning rate, identifying parts of tracks that exhibit turning behavior above
        a specified threshold. This can be particularly useful for understanding navigational patterns,
        identifying areas with high maneuvering activity, or studying the behavior of moving agents in response
        to environmental features or obstacles.

        This classification is stored in a boolean Code column.

        If turning rate data is not present, you should run Dataset.compute_turning_rate first.

        Args:
            rate (float): The threshold for turning rate (degrees per second).
            agents (list, optional): A list of agent IDs whose tracks will be analyzed for turning behavior. If None,
                                    the classification considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified based on turning rate. This allows
                                    for the focused analysis of certain tracks. If None, the operation applies to tracks
                                    related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=18, then 
                                 the output column is Code18. Defaults to 18. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying turning tracks'.

        Returns:
            self: The Dataset instance.
        """
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        This is simply a combination of Dataset.classify_in_polygon and Dataset.classify_speed.
        This classifies points on tracks if they meet or don't meet a speed threshold inside of a given polygon.

        Please see their documentation for more details.

        Args:
            polygon: The polygon must be a shapely Polygon, or a Nx2 array representing the outline of a polygon.
            speed (float): The speed threshold used for classification.
            higher (bool, optional): If True, tracks or agents with speeds higher than the specified threshold will
                                    be classified. Defaults to True.
            lower (bool, optional): If True, tracks or agents with speeds lower than the specified threshold will
                                    be classified. Defaults to False.
            agents (list, optional): A list of agent IDs to be included in the speed classification. If None,
                                    the classification considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified by speed. This allows for selective
                                    analysis of certain tracks. If None, the operation applies to tracks related to the
                                    specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults
                                    to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=19, then 
                                 the output column is Code19. Defaults to 19. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks by speed threshold in polygon',
                                which can be customized for more specific descriptions or contexts of the classification.

        Returns:
            self: The Dataset instance.
        """
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
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Classifies tracks if they do a trip between two polygons (either direction).

        The track must start/end in poly1/poly2. Only the first and last points on the track are used for
        this classification.
         
        If a track meets this condition, a boolean Code column will be added with True for every row in the track data.

        poly1 and poly2 must be shapely Polygons or Nx2 numpy arrays representing the outline of a polygon.

        Args:
            poly1: shapely Polygon or Nx2 numpy array of polygon exterior.
            poly2: shapely Polygon or Nx2 numpy array of polygon exterior.
            agents (list, optional): A list of agent IDs to be considered for trip classification. If None, the
                                    classification is applied to tracks associated with all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified based on the defined trip. This
                                    allows for selective analysis of certain tracks. If None, the operation applies
                                    to tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults
                                    to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=20, then 
                                 the output column is Code20. Defaults to 20. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks by trip'.

        Returns:
            self: The Dataset instance.
        """
        #get polygons
        poly1, edges1 = utils.format_polygon(poly1)
        poly2, edges2 = utils.format_polygon(poly2)
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Classifies tracks based on whether they touch or intersect a specified geometric object. 
        
        The geom kwarg must be a shapely LineString, MultiLineString, Polygon, or MultiPolygon.

        If a given track touches this object, the resulting boolean Code column will be True for
        every row in the track data.
        
        This function is useful for identifying movements that come into contact with geographic 
        features or defined areas, supporting analyses related to boundary crossings, 
        interactions with areas of interest, or proximity to specific objects.

        Args:
            geom: a shapely LineString, MultiLineString, Polygon, or MultiPolygon to check tracks against
            agents (list, optional): A list of agent IDs whose tracks will be analyzed for touching the specified geometry.
                                    If None, the classification considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be classified based on contact with the geometry.
                                    This allows for selective analysis of certain tracks. If None, the operation applies
                                    to tracks related to the specified agents or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=21, then 
                                 the output column is Code21. Defaults to 21. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks touching object'.

        Returns:
            self: The Dataset instance.
        """
        msg = 'geom must be shapely LineString, MultiLineString, Polygon, or MultiPolygon'
        assert isinstance(geom, (LineString, 
                                 MultiLineString, 
                                 Polygon, 
                                 MultiPolygon)), msg
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
        """
        Classifies points along tracks where the agents have stopped, based on a specified speed threshold and minimum stop duration. 
        
        If points along tracks meet the stopping thresholds, the resulting boolean Code column will be True, otherrwise False.

        The stop_threshold is in the same units as the Dataset CRS, i.e. degrees for geographic or meters for UTM, etc.

        The agent has to be stopped for min_stop_duration for it to be considered a valid stop.

        Args:
            stop_threshold (float): The speed below which an agent is considered to be stopped.
            min_stop_duration (int): The minimum duration (in seconds) that an agent must be stopped for the
                                            stop to be considered valid. Defaults to 1800 seconds (30 minutes).
            agents (list, optional): A list of agent IDs whose tracks will be analyzed for stops. If None, the classification
                                    considers all agents in the dataset.
            tracks (list, optional): A list of specific track IDs to be analyzed for stops. This allows for selective analysis
                                    of certain tracks. If None, the operation applies to tracks related to the specified agents
                                    or all tracks if no agents are specified.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=22, then 
                                 the output column is Code22. Defaults to 22. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying stopped tracks'.

        Returns:
            self: The Dataset instance.
        """
            #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(agents, tracks)))
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
                        code=23,
                        desc='Classifying tracks with custom code',
                        meta='Custom classifier'):
        """
        Applies a custom classification to tracks based on an input pandas DataFrame.

        This DataFrame must have an index consisting of Track IDs, and a "Value" column consisting
        of True/False values to classify the tracks.
        
        This offers flexibility to implement user-defined classification logic into the trackio framework, 
        supporting a wide range of analytical needs from behavior categorization to environmental interaction analysis. 

        Args:
            values (pd.DataFrame): values must be a pd.DataFrame with Track IDs as index, and a "Value" column consisting of booleans.
            ncores (int, optional): The number of processing cores to use for the classification operation. Defaults to 1.
            code (int, optional): A numerical code to be used for the resulting boolean column. E.g. if code=23, then 
                                 the output column is Code23. Defaults to 23. Do not use 0.
            desc (str, optional): A brief description of the operation. Defaults to 'Classifying tracks with custom code'.
            meta (str, optional): A description related to the custom classifier being applied. This gets added
                                 to the Dataset.meta attribute for record keeping.

        Returns:
            self: The Dataset instance.
        """
        #make sure it contains the column and is boolean
        msg = 'values must be a pd.DataFrame with Track IDs as index, and a "Value" column consisting of booleans'
        assert 'Value' in values.columns and values['Value'].dtype == bool, msg
        #get the files to process
        pkl_groups = list(zip(*self._get_files_tracks_to_process(None, values.index)))
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