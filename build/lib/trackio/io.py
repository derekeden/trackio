from .utils import collect_agent_pkls
from .Agent import gen_track_meta

import numpy as np
from shapely.geometry import LineString, MultiLineString
import rasterio as rio
from rasterio.transform import Affine
import os
from osgeo import gdal, gdalconst
from pyproj import CRS

def to_track_gdf(agent, track, tid, code):
    #if just returning whole track
    if code is None:
        linestring = LineString([[x, y] for x,y in zip(track['X'], track['Y'])])
    #if returning split multilinestring of only coded parts
    else:
        mask = track[f'Code{code}']
        diffs = np.diff(mask)
        #if all points meet codemask criteria
        if np.sum(diffs) == 0 and np.all(mask):
            #return whole track
            linestring = LineString([[x, y] for x,y in zip(track['X'], track['Y'])])
        #if only some points meet codemask critiera, split
        else:
            split_ids = np.nonzero(diffs)[0]+1
            xs = np.split(track['X'], split_ids)
            ys = np.split(track['Y'], split_ids)
            masks = np.split(mask, split_ids)
            #skip if all only have 1 ping
            if np.all([len(mask)==1 for mask in masks]):
                return
            #make detached linestrings/multilinestrings
            else:
                linestring = []
                #loop through chunks
                for i in range(len(xs)):
                    #if code is True and more than 1 point
                    if np.all(masks[i]) and len(masks[i]) > 1:
                        _linestring = LineString([[x, y] for x,y in zip(xs[i], ys[i])])
                        linestring.append(_linestring)
                    else:
                        pass
                #somehow still empty?
                if len(linestring) > 0:
                    linestring = MultiLineString(linestring)
                else:
                    return
    #make the row for the geodataframe
    meta = agent.agent_meta.copy()
    #delete this entry
    del meta['Tracks']
    meta.update(**agent.track_meta[tid].copy())
    meta['geometry'] = linestring
    return [meta]

def to_segment_gdf(agent,
                   track,
                   tid,
                   code,
                   method):
    #get track id
    track_id = agent.track_meta[tid]['Track ID']
    #initialize rows
    rows = []
    #loop over track points
    for i in range(len(track)-1):
        subtrack = track.iloc[i:i+2]
        #check if it meets the codemask
        if code is not None:
            #if neither part of segment passes
            if not subtrack[f'Code{code}'].any():
                continue        
        #make segment linestring
        linestring = LineString(subtrack[['X','Y']])
        #make segment id
        segment_id = f'{track_id}_S{i}'
        #make meta
        meta = agent.agent_meta.copy()
        #delete this entry
        del meta['Tracks']
        #add track meta
        tmeta = gen_track_meta(subtrack)
        if method in ['forward', 'backward']:
            if method == 'forward':
                index = 0
            else:
                index = 1
            for col in subtrack.columns:
                if col not in ['Time','X','Y']:
                    tmeta[col] = subtrack.iloc[index][col]
        else:
            for_tmeta = subtrack.mean().to_frame().T.astype(subtrack.dtypes)
            for col in for_tmeta.columns:
                if col not in ['Time','X','Y']:
                    tmeta[col] = for_tmeta.iloc[0][col]
        #see if still passing the code requirement after interpolation method
        if code is not None:
            if not tmeta[f'Code{code}']:
                continue
        #finish the meta
        meta.update(**tmeta)
        #add geometry and segment ID
        meta['Segment ID'] = segment_id
        meta['geometry'] = linestring
        rows.append(meta)        
    return rows

def to_gdf(code,
           segments,
           method,
           args):
    rows = []
    #split the agrs
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        #skip tracks with only 1 point
        if len(track) < 2:
            continue
        #skip if doesnt have the code
        if code is not None:
            #if 1 or less points
            if len(track[track[f'Code{code}']]) < 2:
                continue
        if segments:
            meta = to_segment_gdf(agent, track, tid, code, method)
        else:
            meta = to_track_gdf(agent, track, tid, code)
        #if gdf row(s) returned
        if meta is not None:
            rows.extend(meta)
    return rows

def to_df(code,
          args):
    dfs = []
    #split the agrs
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        track_id = agent.track_meta[tid]['Track ID']
        #get rid of codes
        if code is not None:
            track = track[track[f'Code{code}']].copy() #copy? wtf?
        #if nothing left
        if len(track) < 1:
            continue
        #make point ids
        track.index = [f'{track_id}_P{i}' for i in range(len(track))]
        #add agent meta
        meta = agent.meta
        for key,val in meta.items():
            track.loc[:, key] = val
        #append to list
        dfs.append(track)
    return dfs

def create_blank_raster(out_file, crs,
                  grid={'x0':0, 'y0':0, 'nx':1, 'ny':1, 'dx':1, 'dy':1}):
    assert out_file.lower().endswith('.tif'), 'out_file must be a .tif file'
    profile = {'count': 1,
               'crs': CRS(crs),
               'driver': 'GTiff',
               'dtype': 'float32',
               'height': grid['ny'],
               'interleave': 'band',
               'nodata': None,
               'tiled': False,
               'transform': Affine(grid['dx'], 0.0, grid['x0'],
                                   0.0, grid['dy'], grid['y0']),
               'width': grid['nx']}
    data = np.zeros((grid['ny'], grid['nx']))
    with rio.open(out_file, 'w', **profile) as f:
        f.write(data, 1)
    print(f'Raster created and saved to {out_file}!')
    return

# def rasterize_gdf(gdf, out_file, attr=None, 
#                   grid={'x0':0, 'y0':0, 'nx':1, 'ny':1, 'dx':1, 'dy':1}):
#     out_file = os.path.abspath(out_file)
#     if attr is not None:
#         shp = gdal.OpenEx(gdf[[attr,'geometry']].to_json(), gdal.OF_VECTOR)
#     else:
#         shp = gdal.OpenEx(gdf[['geometry']].to_json(), gdal.OF_VECTOR)
#     output = gdal.GetDriverByName('GTiff').Create(os.path.abspath(out_file), grid['nx'], grid['ny'], 1, gdal.GDT_Float32)
#     output.SetGeoTransform([grid['x0'],
#                             grid['dx'],
#                             0,
#                             grid['y0'],
#                             0,
#                             grid['dy']])
#     output.SetProjection(gdf.crs.to_proj4())
#     output.GetRasterBand(1).SetNoDataValue(0)
#     if attr is not None:
#         options = gdal.RasterizeOptions(add=True, attribute=attr, allTouched=True)
#     else:
#         options = gdal.RasterizeOptions(add=True, burnValues=1, allTouched=True)
#     gdal.Rasterize(output, shp, options=options)
#     output = None
#     print(f'Raster written to {out_file}')

def rasterize(shp_file, ras_file, out_file, attribute=None):
    #make abspath
    out_file = os.path.abspath(out_file)
    ras_file = os.path.abspath(ras_file)
    shp_file = os.path.abspath(shp_file)    
    #make gdal raster inputs
    ras = gdal.Open(ras_file, gdalconst.GA_ReadOnly)
    geo_transform = ras.GetGeoTransform()
    x_res = ras.RasterXSize
    y_res = ras.RasterYSize
    output = gdal.GetDriverByName('GTiff').Create(out_file, x_res, y_res, 1, gdal.GDT_Float32)
    output.SetGeoTransform(geo_transform)
    output.SetProjection(ras.GetSpatialRef().ExportToWkt())
    output.GetRasterBand(1).SetNoDataValue(0)  
    if attribute is None:
        #rasterize counts
        gdal.Rasterize(output, 
                       shp_file, 
                       options=gdal.RasterizeOptions(add=True, 
                                                     burnValues=1,
                                                     allTouched=True)) 
        print(f'Track Counts written to {out_file}')
    else:  
        #rasterize attribute
        gdal.Rasterize(output, 
                       shp_file, 
                       options=gdal.RasterizeOptions(add=True, 
                                                     attribute=attribute,
                                                     allTouched=True))
        print(f'Track {attribute} written to {out_file}')
    output = None
    
################################################################################