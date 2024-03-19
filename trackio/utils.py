
################################################################################

import pickle as pkl
import numpy as np
from shapely.geometry import (Polygon, 
                              MultiPolygon, 
                              LineString, 
                              MultiLineString, 
                              Point, 
                              MultiPoint)
from scipy.spatial import Voronoi
from pyproj import CRS
import multiprocessing as mp
from functools import partial
from tqdm import tqdm; GREEN = "\033[92m"; ENDC = "\033[0m" #for tqdm bar
import pandas as pd
import shapely
import geopandas as gp

################################################################################

def append_pkl(out_file, pkl_obj):
    #appends a serialized chunk to the pickle file
    with open(out_file, "ab") as p:
        pkl.dump(pkl_obj, p)

def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pkl.load(f)

def save_pkl(pkl_file, pkl_obj):
    with open(pkl_file, 'wb') as f:
        pkl.dump(pkl_obj, f)

def collect_agent_pkls(pkl_files):
    #loop over pkl files
    i = 0
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as p:
            #read until last chunk.. important!!!
            while True:
                try:
                    _tmp = pkl.load(p)
                    if _tmp is not None:
                        if i == 0:
                            agent = _tmp #these are pickled Agent objects
                        else:
                            agent.append_data.append(_tmp.data) #append the unsplit points
                        i += 1
                except EOFError:
                    break
    return agent

def flatten(l):
    return [item for sublist in l for item in sublist]

def flatten_dict_unique(_out):
    out = {}
    for o in _out:
        for key in o.keys():
            if key not in out.keys():
                out[key] = o[key]
            else:
                out[key].extend(o[key])
    for key in out.keys():
        out[key] = np.unique(out[key]).tolist()
    return out

def std_meta():
    meta = {'X':'degrees',
            'Y':'degrees',
            'CRS': CRS(4326),
            'Static Data': [],
            'Dynamic Data': ['X','Y']}
    return meta

def format_input(inp, typ):
    if isinstance(inp, str):
        if typ == int:
            numeric = ''.join([l for l in inp if l.isnumeric()])
            if len(numeric) == 0:
                inp = typ(0)
            else:
                inp = typ(numeric)
        elif typ == float:
            numeric = ''.join([l for l in inp if (l.isnumeric() or l == '.')])
            if len(numeric) == 0 or all([l == '.' for l in numeric]):
                inp = typ(0)
            else:
                inp = typ(numeric)
    elif np.isnan(inp):
        inp = typ(0)
    else:
        inp = typ(inp)
    return inp

def format_polygon(poly):
    if isinstance(poly, Polygon):
        poly = np.array(poly.exterior.xy).T
        edges = [(i,i+1) for i in range(len(poly)-1)]
    elif isinstance(poly, MultiPolygon):
        edges = []
        polys = []
        j = 0
        for geom in poly.geoms:
            p = geom.exterior.xy
            polys.append(np.array(p).T)
            edges.extend([(i+j,i+j+1) for i in range(len(p[0])-1)])
            j += len(p[0])
        poly = np.concatenate(polys)
    elif isinstance(poly, np.ndarray):
        edges = [(i,i+1) for i in range(len(poly)-1)]
        pass
    else:
        raise Exception('Polygon must be shapely polygon or 2D numpy array of polygon exterior')
    return poly, edges

def pool_caller(func, partials, iterable, desc, ncores):
    #join func and kwargs
    map_func = partial(func, *partials)
    #if only 1 core
    out = []
    if ncores == 1:
        for i in tqdm(iterable, total=len(iterable), desc=GREEN+desc+ENDC, colour='GREEN'):
            out.append(map_func(i))
    #if in parallel
    else:
        with mp.Pool(ncores) as pool:
            out = pool.map(map_func, 
                        tqdm(iterable, 
                             total=len(iterable), 
                             desc=GREEN+desc+ENDC,
                             colour='GREEN'))
    return out

def prepare_polylines(shape):
    #convert shape into set of polylines
    polylines = []
    if isinstance(shape, (MultiLineString, 
                        MultiPolygon, MultiPoint)):
        for geom in shape.geoms:
            if isinstance(geom, (LineString, Point)):
                polylines.append(np.array(geom.xy).T)
            else:
                polylines.append(np.array(geom.exterior.xy).T)
    else:
        if isinstance(shape, (LineString, Point)):
            polylines.append(np.array(shape.xy).T)
        else: #Polygon
            polylines.append(np.array(shape.exterior.xy).T)
    return polylines

def split_list(lst, n):
    # Using list comprehension to generate chunks
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def gdf_to_df(gdf):
    #prep it into a dataframe
    rows = []
    for i in range(len(gdf)):
        #get master row
        _row = gdf.iloc[i].to_dict()
        #track coords
        x, y = _row['geometry'].xy
        #dummy times, 1 second, all overlapping
        times = pd.date_range(0,1e9, periods=len(x))
        for x,y,t in zip(x, y, times):
            row = _row.copy()
            row['Time'] = t
            row['X'] = x
            row['Y'] = y
            rows.append(row)
    return pd.DataFrame(rows)

def longitudinal_slices(start, 
                        end, 
                        n_slices, 
                        spacing, 
                        dist):
    #if n_slices instead of spacing
    if n_slices > 0:
        if n_slices == 1:
            #make longitudinal slices
            long = np.array([0])
        else:
            #make longitudinal slices
            long = np.linspace(0, dist, n_slices)
    #if spacing
    else:
        #make longitudinal slices
        long = np.concatenate([np.arange(0, 
                                        dist, 
                                        spacing), 
                            [dist]])
    #turn into coordinates
    xy = []
    for i in range(len(long)):
        dx = (end[0]-start[0]) * long[i]/dist
        dy = (end[1]-start[1]) * long[i]/dist
        xy.append((start[0]+dx, start[1]+dy))
    xy = np.array(xy)
    return long, xy

def range_to_edges(global_min, global_max, bins):
    #all entries the same
    if global_min == global_max:
        edges = np.array([global_min, global_min+bins])
    #distribution of entries
    else:
        #if both +/-
        if np.sign(global_min) == np.sign(global_max):
            sign = np.sign(global_min)
            abs_max = np.abs([global_min, global_max]).max() 
            edges = np.sort(np.arange(0, abs_max+bins, bins)*sign)
        #if + and -  
        else:
            edges1 = np.arange(0, global_max+bins, bins)
            edges2 = -np.arange(0, np.abs(global_min)+bins, bins)
            edges = np.sort(np.unique(np.concatenate([edges1, edges2])))
    return edges

def first_nonzero(lst):
    out = 0
    for l in lst:
        if l != 0:
            out = l
            break
    return out
                
def NN_idx2(query, coords):
    dx = coords[:,:,0] - query[0]    
    dy = coords[:,:,1] - query[1]
    dr = dx**2 + dy**2        
    idx2 = np.unravel_index(np.argmin(dr), dr.shape)
    return idx2

def create_voronoi(centroids, buffer=500000):
    #make buffer points
    centroid = Point(centroids[['X','Y']].values.mean(axis=0))
    buffer_pts = np.array(centroid.buffer(buffer, resolution=1000).exterior.xy).T
    #combine centroids + buffer points
    pts = np.vstack([centroids[['X','Y']].values, buffer_pts])
    #make voronoi diagram
    vor = Voronoi(pts)
    #convert to poylgons
    lines = [
        LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]
    polys = list(shapely.ops.polygonize(lines))
    #convert to geodataframe, clip to domain extent
    voronoi = gp.GeoDataFrame(geometry=polys, crs=centroids.crs)
    #add centroids to dataframe
    voronoi['Code'] = np.array(list(range(len(voronoi))))+1
    voronoi['X'] = np.nan
    voronoi['Y'] = np.nan
    for i in range(len(voronoi)):
        for p in centroids.geometry:
            if voronoi.geometry.iloc[i].intersects(p):
                voronoi.iat[i, voronoi.columns.get_loc('X')] = p.x
                voronoi.iat[i, voronoi.columns.get_loc('Y')] = p.y
    return voronoi

################################################################################