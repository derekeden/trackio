################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

################################################################################

must_cols = ['Time','X','Y']

class Agent:
    def __init__(self, 
                 id='agent',
                 meta={},
                 data=pd.DataFrame({c:[] for c in must_cols})):
        #set data vars      
        self._data = data
        self.append_data = []
        # self.split_ids = []  
        #set meta vars
        self.meta = {}
        self.meta.update(meta)
        self.meta['Agent ID'] = id
        #init dicts
        self.tracks = {}   
        self.agent_meta = {}
        self.track_meta = {}   
        
    ############################################################################
    #ATTRIBUTES
    ############################################################################
    
    @property
    def data(self):
        if self._data is None:
            return pd.concat(self.tracks.values()).reset_index(drop=True)
        else:
            return self._data
        
    ############################################################################
    #METHODS
    ############################################################################
        
    def split_tracks_spatiotemporal(self, 
                                    time, 
                                    distance,
                                    tracks=[],
                                    method=0):
        #if only specific tracks
        if len(tracks) > 0:
            old_track_ids = [f"{self.meta['Agent ID']}_T{i}" for i in range(len(self.tracks))]
            keep_track_ids = [tid.rsplit('_', 1)[1] for tid in old_track_ids if tid not in tracks]
            keep_tracks = [self.tracks[idx] for idx in keep_track_ids]
            for tid in keep_track_ids:
                self.tracks.pop(tid)
        #grab additional data for this agent group
        if len(self.append_data) > 0:
            data = pd.concat([self.data, *self.append_data]).copy()
            self.append_data = []
        else:
            data = self.data.copy()
        if method == 0:
            #sort the data by time and get uniq
            data = data.drop_duplicates(subset='Time')
            data = data.sort_values(by='Time').reset_index(drop=True)
            #split the data by time differences
            time_splits = np.diff(data['Time']).astype(np.int64) * 1e-9 > time #nanoseconds to seconds
            #split the data by distance
            dists = np.sqrt(np.diff(data['X'])**2 + np.diff(data['Y'])**2) 
            dist_splits = dists > distance
            #could be either or OR both need short-circuit
            splits = time_splits | dist_splits
            split_ids = np.nonzero(splits)[0] + 1
            # self.split_ids = split_ids
            #split into track dataframes
            idxs = np.split(range(len(data)), split_ids)
        else:
            #sort the data by time
            data = data.sort_values(by='Time').reset_index(drop=True) 
            idxs = [[0]]
            #loop over the data sequentially to account for overlapping data (i.e. wrong identifier for 2+ tracks)
            for i in range(1, len(data)):
                new_track = True
                #for each already established track
                for track_idx in idxs:
                    old_id = track_idx[-1]
                    dt = (data.iloc[i]['Time'] - data.iloc[old_id]['Time']).total_seconds()
                    dy = data.iloc[i]['Y'] - data.iloc[old_id]['Y'] 
                    dx = data.iloc[i]['X'] - data.iloc[old_id]['X'] 
                    dist = (dx**2+dy**2)**0.5
                    #if still on this track
                    if dt <= time and dist <= distance:
                        track_idx.append(i)
                        new_track = False
                        break
                #if it didn't get assigned to any existing tracks
                if new_track:  
                    idxs.append([i])
        #reset tracks
        self.tracks = {}   
        #init new track meta
        track_meta = {}
        for i, idx in enumerate(idxs):
            #get split track
            tdf = data.iloc[idx].copy()
            #save track data
            self.tracks[f'T{i}'] = tdf.reset_index(drop=True)
            track_meta[f'T{i}'] = gen_track_meta(tdf)
            track_meta[f'T{i}']['Track ID'] = f"{self.meta['Agent ID']}_T{i}"
        #if only specific tracks were split, re-add the old ones that weren't touched
        nstart = len(self.tracks)
        if len(tracks) > 0:
            for i in range(len(keep_tracks)):
                j = i + nstart
                #save track data
                self.tracks[f'T{j}'] = keep_tracks[i]
                track_meta[f'T{j}'] = gen_track_meta(keep_tracks[i])
                track_meta[f'T{j}']['Track ID'] = f"{self.meta['Agent ID']}_T{j}"
        #update the track meta
        self.track_meta = track_meta
        #add the agent meta
        self.agent_meta = gen_agent_meta(self)
        #delete the original data as it's now stored in tracks, don't double it
        self._data = None
        return self  
      
    def split_tracks_by_data(self, 
                             data_col,
                             tracks=[]):
        #if only specific tracks
        if len(tracks) > 0:
            old_track_ids = [f"{self.meta['Agent ID']}_T{i}" for i in range(len(self.tracks))]
            keep_track_ids = [tid.rsplit('_', 1)[1] for tid in old_track_ids if tid not in tracks]
            keep_tracks = [self.tracks[idx] for idx in keep_track_ids]
            for tid in keep_track_ids:
                self.tracks.pop(tid)
            #loop over tracks and split
            split_tracks = []
            for tid in self.tracks.keys():
                track = self.tracks[tid]
                #split the data by changes in data column
                splitter = track[data_col].values
                splits = np.diff(splitter)
                split_ids = np.nonzero(splits)[0]+1
                idxs = np.split(range(len(track)), split_ids)
                for idx in idxs:
                    split_tracks.append(track.iloc[idx].copy().reset_index(drop=True))
        #if from points
        else:
            split_tracks = []
            #grab additional data for this agent group
            if len(self.append_data) > 0:
                data = pd.concat([self.data, *self.append_data]).copy()
                self.append_data = []
            else:
                data = self.data.copy()
            #split the data by changes in data column
            splitter = data[data_col].values
            splits = np.diff(splitter)
            split_ids = np.nonzero(splits)[0]+1
            idxs = np.split(range(len(data)), split_ids)
            for idx in idxs:
                split_tracks.append(data.iloc[idx].copy().reset_index(drop=True))
        #reset tracks
        self.tracks = {}   
        #init new track meta
        track_meta = {}
        for i,tdf in enumerate(split_tracks):
            #save track data
            self.tracks[f'T{i}'] = tdf
            track_meta[f'T{i}'] = gen_track_meta(tdf)
            track_meta[f'T{i}']['Track ID'] = f"{self.meta['Agent ID']}_T{i}"
        #if only specific tracks were split, re-add the old ones that weren't touched
        nstart = len(self.tracks)
        if len(tracks) > 0:
            for i in range(len(keep_tracks)):
                j = i + nstart
                #save track data
                self.tracks[f'T{j}'] = keep_tracks[i]
                track_meta[f'T{j}'] = gen_track_meta(keep_tracks[i])
                track_meta[f'T{j}']['Track ID'] = f"{self.meta['Agent ID']}_T{j}"
        #update the track meta
        self.track_meta = track_meta
        #add the agent meta
        self.agent_meta = gen_agent_meta(self)
        #delete the original data as it's now stored in tracks, don't double it
        self._data = None
        return self 
        
    def split_tracks_kmeans(self,
                            n_clusters=range(1,4),
                            feature_cols=['X','Y'],
                            tracks=[],
                            return_inertia=False,
                            optimal_method='davies-bouldin',
                            **kwargs):        
        #if only specific tracks
        if len(tracks) > 0:
            old_track_ids = [f"{self.meta['Agent ID']}_T{i}" for i in range(len(self.tracks))]
            keep_track_ids = [tid.rsplit('_', 1)[1] for tid in old_track_ids if tid not in tracks]
            keep_tracks = [self.tracks[idx] for idx in keep_track_ids]
            for tid in keep_track_ids:
                self.tracks.pop(tid)
        #grab additional data for this agent group
        if len(self.append_data) > 0:
            data = pd.concat([self.data, *self.append_data]).copy()
            self.append_data = []
        else:
            data = self.data.copy()
        #make old copy
        old_data = data.dropna(subset=feature_cols)
        old_data = old_data.sort_values(by='Time').reset_index(drop=True)
        #reduce to features, drop nan values
        data = old_data[feature_cols].copy()
        #convert time to seconds
        if 'Time' in feature_cols:
            data['Time'] = data['Time'].astype(np.int64) * 1e-9
        #check if encoding is needed for non numeric columns
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                # Define a transformer for the one-hot encoding
                transformer = ColumnTransformer(transformers=[
                    (col, OneHotEncoder(), [col])
                ], remainder='passthrough')
                #transform the data
                data = transformer.fit_transform(data)
        #standardize the features
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        #loop over the k-values
        n_clusters = sorted(n_clusters)
        clusters = []
        inertia = []
        daviesbouldin = []
        silhouette = []
        error_rows = []
        for k in n_clusters:
            kmeans = KMeans(n_clusters=k, **kwargs)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)
            s = silhouette_score(data, kmeans.labels_)
            silhouette.append(s)
            db = davies_bouldin_score(data, kmeans.labels_)
            daviesbouldin.append(db)
            clusters.append(kmeans.labels_)
            error_rows.append({'Agent ID': self.meta['Agent ID'],
                               'n_clusters': k,
                               'inertia': kmeans.inertia_,
                               'davies-bouldin': db,
                               'silhouette': s})
        if len(n_clusters) == 1:
            optimal_id = 0
        else:
            if optimal_method == 'davies-bouldin':
                optimal_id = int(np.argmin(daviesbouldin))
            elif optimal_method == 'silhouette':
                optimal_id = int(np.argmax(silhouette))
            else:
                kneedle = KneeLocator(n_clusters,
                            inertia, 
                            curve='convex', 
                            direction='decreasing')
                optimal_id = kneedle.knee
        assert isinstance(optimal_id, int), 'could not find an optimal n_clusters value'
        #get labels
        labels = clusters[optimal_id]
        uniq_labels = np.unique(labels)
        #reset tracks
        self.tracks = {}   
        #init new track meta
        track_meta = {}
        #loop over tracks
        for i,label in enumerate(uniq_labels):
            mask = labels == label
            #get split track
            tdf = old_data.loc[mask].copy()
            #save track data
            self.tracks[f'T{i}'] = tdf.reset_index(drop=True)
            track_meta[f'T{i}'] = gen_track_meta(tdf)
            track_meta[f'T{i}']['Track ID'] = f"{self.meta['Agent ID']}_T{i}"
        #if only specific tracks were split, re-add the old ones that weren't touched
        nstart = len(self.tracks)
        if len(tracks) > 0:
            for i in range(len(keep_tracks)):
                j = i + nstart
                #save track data
                self.tracks[f'T{j}'] = keep_tracks[i]
                track_meta[f'T{j}'] = gen_track_meta(keep_tracks[i])
                track_meta[f'T{j}']['Track ID'] = f"{self.meta['Agent ID']}_T{j}"
        #update the track meta
        self.track_meta = track_meta
        #add the agent meta
        self.agent_meta = gen_agent_meta(self)
        #delete the original data as it's now stored in tracks, don't double it
        self._data = None
        if return_inertia:
            return self, error_rows
        else:
            return self

    def split_tracks_dbscan(self,
                            feature_cols=['X','Y'],
                            tracks=[],
                            eps=0.5,
                            min_samples=2,
                            **kwargs):       
            #if only specific tracks
            if len(tracks) > 0:
                old_track_ids = [f"{self.meta['Agent ID']}_T{i}" for i in range(len(self.tracks))]
                keep_track_ids = [tid.rsplit('_',1)[1] for tid in old_track_ids if tid not in tracks]
                keep_tracks = [self.tracks[idx] for idx in keep_track_ids]
                for tid in keep_track_ids:
                    self.tracks.pop(tid)
            #grab additional data for this agent group
            if len(self.append_data) > 0:
                data = pd.concat([self.data, *self.append_data]).copy()
                self.append_data = []
            else:
                data = self.data.copy()
            #make old copy
            old_data = data.dropna(subset=feature_cols)
            old_data = old_data.sort_values(by='Time').reset_index(drop=True)
            #reduce to features, drop nan values
            data = old_data[feature_cols].copy()
            #convert time to seconds
            if 'Time' in feature_cols:
                data['Time'] = data['Time'].astype(np.int64) * 1e-9
            #check if encoding is needed for non numeric columns
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    # Define a transformer for the one-hot encoding
                    transformer = ColumnTransformer(transformers=[
                        (col, OneHotEncoder(), [col])
                    ], remainder='passthrough')
                    #transform the data
                    data = transformer.fit_transform(data)
            #standardize the features
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            #cluster the data
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            #get labels
            labels = dbscan.fit(data).labels_
            uniq_labels = np.unique(labels)
            #reset tracks
            self.tracks = {}   
            #init new track meta
            track_meta = {}
            #loop over tracks
            for i,label in enumerate(uniq_labels):
                mask = labels == label
                #get split track
                tdf = old_data.loc[mask].copy()
                #save track data
                self.tracks[f'T{i}'] = tdf.reset_index(drop=True)
                track_meta[f'T{i}'] = gen_track_meta(tdf)
                track_meta[f'T{i}']['Track ID'] = f"{self.meta['Agent ID']}_T{i}"
            #if only specific tracks were split, re-add the old ones that weren't touched
            nstart = len(self.tracks)
            if len(tracks) > 0:
                for i in range(len(keep_tracks)):
                    j = i + nstart
                    #save track data
                    self.tracks[f'T{j}'] = keep_tracks[i]
                    track_meta[f'T{j}'] = gen_track_meta(keep_tracks[i])
                    track_meta[f'T{j}']['Track ID'] = f"{self.meta['Agent ID']}_T{j}"
            #update the track meta
            self.track_meta = track_meta
            #add the agent meta
            self.agent_meta = gen_agent_meta(self)
            #delete the original data as it's now stored in tracks, don't double it
            self._data = None
            return self

################################################################################

#generate the metadata
def gen_track_meta(tdf):
    first = tdf.iloc[0]
    last = tdf.iloc[-1]
    t0 = first['Time']
    t1 = last['Time']
    dts = np.diff(tdf['Time']).astype("timedelta64[s]").astype(np.int64)
    dxs = (np.diff(tdf['X'])**2 + np.diff(tdf['Y'])**2)**0.5
    tmeta = {
        "Track Length": ((tdf['X'].diff()**2 + tdf['Y'].diff()**2)**0.5).sum(),
        "npoints": len(tdf),
        "Start Time": t0,
        "End Time": t1,
        "Duration": int((t1-t0).total_seconds()),
        "Year": int(t1.year),
        "Month": int(t1.month),
        "Xmin": tdf['X'].min(),
        "Xmax": tdf['X'].max(),
        "Ymin": tdf['Y'].min(),
        "Ymax": tdf['Y'].max(),
        "Xstart": first['X'],
        "Ystart": first['Y'],
        "Xend": last['X'],
        "Yend": last['Y'],
        "Effective Distance": ((first['X'] - last['X']) ** 2 + (first['Y'] - last['Y']) ** 2) ** 0.5,
        # "Draft": first['Draft'],
        # "Min Speed": tdf['Speed'].min(),
        # "Mean Speed": tdf['Speed'].mean(),
        # "Max Speed": tdf['Speed'].max(),
        # "Max Acceleration": tdf['Acceleration'].max(),
        # "Max Decceleration": tdf['Acceleration'].min(),
        # "Max Turning Rate": tdf['Turning Rate'].max(),
        "Min Temporal Resolution": np.nanmean(dts) if len(dts) > 0 else 0,
        "Mean Temporal Resolution": np.nanmean(dts) if len(dts) > 0 else 0,
        "Max Temporal Resolution": np.nanmax(dts) if len(dts) > 0 else 0,
        "Min Spatial Resolution": np.nanmin(dxs) if len(dxs) > 0  else 0,
        "Mean Spatial Resolution": np.nanmean(dxs) if len(dxs) > 0  else 0,
        "Max Spatial Resolution": np.nanmax(dxs) if len(dxs) > 0  else 0,
        **tdf.filter(like='Code').any(axis=0).to_dict()
        }
    return tmeta

def gen_agent_meta(self):
    #grab some basic metadata
    vdf = self.data
    vmeta = {**self.meta,
        "npoints": len(self.data),
        "ntracks": len(self.tracks),
        "Xmin": vdf['X'].min(),
        "Xmax": vdf['X'].max(),
        "Ymin": vdf['Y'].min(),
        "Ymax": vdf['Y'].max(),
        "Start Time": vdf['Time'].min(),
        "End Time": vdf['Time'].max(),
        **vdf.filter(like='Code').any(axis=0).to_dict()
        }
    return vmeta