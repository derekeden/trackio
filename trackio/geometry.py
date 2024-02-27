################################################################################

from .utils import read_pkl, save_pkl, collect_agent_pkls

import numpy as np
from pyproj import CRS, Geod
import os
import pandas as pd
from shapely.geometry import Point, LineString
from inpoly import inpoly2
from more_itertools import consecutive_groups

#for dividing by zero in lin alg equations - returns nan anyways, but annoying
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

################################################################################

def interpolate_dynamic_data(track, newx, oldx):
    #setup dict
    out = {}
    #get dtypes
    dtypes = track.dtypes
    #loop over columns
    for col in track.columns:
        #if numeric
        if pd.api.types.is_numeric_dtype(dtypes[col]):
            out[col] = np.interp(newx, oldx, track[col].values)
        #if datetime
        elif col == 'Time':
            out[col] = pd.to_datetime(np.interp(newx, 
                                                oldx, 
                                                track[col].astype(np.int64)))
        #if boolean
        elif pd.api.types.is_bool_dtype(dtypes[col]):
            out[col] = np.interp(newx, oldx, track[col].values).astype(bool)
        #if string
        else:
            #map to ints
            uniq = track[col].unique()
            mapper = {u:i for i,u in enumerate(uniq)}
            inv_mapper = dict(zip(mapper.values(), mapper.keys()))
            dat = track[col].apply(lambda x: mapper.get(x)).values
            #interpolate the ints
            newdat = np.interp(newx, oldx, dat).round().astype(int)
            out[col] = list(map(lambda x: inv_mapper.get(x), newdat))
    #return dataframe
    return pd.DataFrame(out)

def points_to_coursing(x, y, crs, method='forward'):
    if CRS(crs).is_geographic:
        geod = Geod(ellps="WGS84")
        courses, _, _ = geod.inv(x[:-1], y[:-1], x[1:], y[1:])     
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        courses = np.degrees(np.arctan2(dx, dy))
    if method == 'forward':
        courses = np.hstack([courses, courses[-1:]])
    elif method == 'backward':
        courses = np.hstack([courses[:1], courses])
    else:
        c1 = np.hstack([courses, courses[-1:]])
        c2 = np.hstack([courses[:1], courses])
        courses = (c1+c2)/2
    return courses % 360
        
def decimate(points, epsilon):
    # get the start and end points
    start = np.tile(np.expand_dims(points[0], axis=0), (points.shape[0], 1))
    end = np.tile(np.expand_dims(points[-1], axis=0), (points.shape[0], 1))
    # find distance from other_points to line formed by start and end
    dist_point_to_line = np.abs(np.cross(end - start, points - start, axis=-1)) / np.linalg.norm(end - start, axis=-1)
    # get the index of the points with the largest distance
    max_idx = np.argmax(dist_point_to_line)
    max_value = dist_point_to_line[max_idx]
    result = []
    if max_value > epsilon:
        partial_results_left = decimate(points[:max_idx+1], epsilon)
        result += [list(i) for i in partial_results_left if list(i) not in result]
        partial_results_right = decimate(points[max_idx:], epsilon)
        result += [list(i) for i in partial_results_right if list(i) not in result]
    else:
        result += [points[0], points[-1]]
    pass
    return np.array(result)

def characteristic_indices(x, y, time, speed, coursing,
                            stop_threshold=0.15,
                            turn_threshold=22.5,
                            min_distance=500,
                            max_distance=20000,
                            min_stop_duration=1800):
    #get the diffs
    dx = (np.diff(x)**2 + np.diff(y)**2)**0.5
    dturn = np.diff(coursing)
    #find the stops
    stops = np.nonzero(speed <= stop_threshold)[0]
    #group the stops
    stop_ids = [list(g) for g in consecutive_groups(stops)]
    stop_ids = [[g[0], g[-1]] for g in stop_ids]
    #remove stops less than threshold
    for i in range(len(stop_ids)):
        stop = stop_ids[i]
        if (time[stop[1]] - time[stop[0]]).total_seconds() <= min_stop_duration:
            stop_ids[i] = stop[1:]
    #combine end pts, stop pts
    stop_ids = [[0]] + stop_ids + [[len(time)-1]]
    #loop over gaps between stops
    keeps = []
    for sid in range(len(stop_ids)-1):
        #get last point of current stop, first point of next - this is the segment i:j
        stop_groups = stop_ids[sid], stop_ids[sid+1]
        i = stop_groups[0][-1]
        j = stop_groups[1][0]
        #iterate through the segment
        while i < j:
            #get idx along segment at which max_distance met
            dxs = dx[i:j].copy()
            dxs[0] = 0
            dist_idx = sum(np.cumsum(dxs) < max_distance) + i
            #get idx along segment at which turn_threshold is met
            dturns = dturn[i:j].copy()
            dturns[0] = 0 
            turn_idx = sum(np.abs(np.cumsum(dturns)) < turn_threshold) + i
            #if small segment and no turns
            if dist_idx == j and turn_idx == i:
                #just connect to the next stop
                break
            else:
                #exceeds max distance before turning
                if dist_idx <= turn_idx and turn_idx != i:
                    #set index to dist index
                    keep = dist_idx
                #turns before max distance is reached
                elif turn_idx <= dist_idx and dist_idx != i:
                    #calc distance along this segment
                    dist = sum(dxs[:turn_idx-i])
                    #if distance less than min seg distance
                    if dist <= min_distance:
                        #short small turns, skip to dist index
                        keep = dist_idx
                    else:
                        #long enough turn, set to turn index
                        keep = turn_idx
                #if longer than max distance and no turns
                elif dist_idx > i:
                    #set to dist index
                    keep = dist_idx
                #if turns but doesnt cross distance threshold
                elif turn_idx > i:
                    #skip to next stop
                    break
                #exceeds distance and turn threshold at same time
                else:
                    #set to either
                    keep = dist_idx
                #reset i, append characteristic id to keeps
                i = keep
                keeps.append(keep)
    #resort the keeps and stop ids
    keeps = sorted(set((keeps + [s for sublist in stop_ids for s in sublist])))
    return keeps

def simplified_stop_indices(x, y, time, speed, 
                           stop_threshold, 
                           min_stop_duration,
                           max_drift_distance):
    #find the stops
    stops = np.nonzero(speed <= stop_threshold)[0]
    #group the stops
    stop_ids = [list(g) for g in consecutive_groups(stops)]
    stop_ids = [s for s in stop_ids if len(s) > 1]
    #remove stops less than duration threshold
    stop_ids = [s for s in stop_ids 
                if (time[s[-1]] - time[s[0]]).total_seconds() >= min_stop_duration]
    #check for long drifting stops, keep points beyond drift threhsold
    keeps = []
    for stop in stop_ids:
        i = stop[0]
        j = stop[-1]
        while i < j:
            keeps.append(i)
            dists = ((x[i:j]-x[i])**2 + (y[i:j]-y[i])**2)**0.5
            dist_idx = np.nonzero(dists >= max_drift_distance)[0]
            if len(dist_idx) == 0:
                i = j
            else:
                i += dist_idx[0] + 1
        keeps.append(j)    
    #reduce but include drift points
    stop_ids = [[s for s in sublist if s in keeps] for sublist in stop_ids]
    #get ids to return
    return_ids = list(range(len(x)))
    for stop in stop_ids:
        for i in range(stop[0], stop[-1]+1):
            if i in stop:
                pass
            else:
                return_ids.pop(return_ids.index(i))
    return return_ids

def reproject_crs(transformer, 
                  out_pth, 
                  args):
    #split args
    pkl_files, tracks = args
    #if there are specific tracks
    if len(tracks) > 0:
        for pkl_file in pkl_files:
            refresh = False
            agent = read_pkl(pkl_file)
            tids = [tid for tid in agent.tracks.keys() 
                    if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
            for tid in tids:
                oldx, oldy = (agent.tracks[tid]['X'].values, 
                              agent.tracks[tid]['Y'].values)
                newx, newy = transformer.transform(oldx, oldy)  
                agent.tracks[tid]['X'] = newx
                agent.tracks[tid]['Y'] = newy
                refresh = True
            #save the agent back
            if refresh:
                out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
                save_pkl(out_file, agent)
    #if entire agents
    else:
        for pkl_file in pkl_files:
            #read agent
            agent = read_pkl(pkl_file)
            #get split ids
            split_ids = np.cumsum([len(t) for t in agent.tracks.values()])
            #get old coordinates
            oldx = agent.data['X']
            oldy = agent.data['Y']
            newx, newy = transformer.transform(oldx, oldy)  
            #if split
            if agent._data is None:
                tx = np.split(newx, split_ids)
                ty = np.split(newy, split_ids)
                for i, tid in enumerate(agent.tracks.keys()):
                    agent.tracks[tid]['X'] = tx[i]
                    agent.tracks[tid]['Y'] = ty[i]
            #if not split (still just points)
            else:
                agent._data['X'] = newx
                agent._data['Y'] = newy
            #save the agent back
            out_file = f'{out_pth}/{os.path.basename(pkl_file)}'
            save_pkl(out_file, agent)

def resample_spacing(spacing, 
                     out_pth,
                     args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over tracks and resample
    refresh = False
    for tid in tids:
        track = agent.tracks[tid]
        #if only 1 point
        if len(track) == 1:
            continue
        #if multiple points
        else:       
            xd = np.diff(track['X'])
            yd = np.diff(track['Y'])
            dist = np.sqrt(xd ** 2 + yd ** 2)
            u = np.cumsum(dist)
            u = np.hstack([[0], u])
            total_dist = u[-1]
            #if total distance of track <= new spacing
            if total_dist <= spacing:
                continue #if total dist less than spacing
            else:
                refresh = True  
                #get the new cumsum
                t = np.hstack([np.arange(0, total_dist, spacing), 
                               [total_dist]])
                #interpolate the other attributes back
                new_track = interpolate_dynamic_data(track, t, u)
                #update the track dataframe
                agent.tracks[tid] = new_track
    if refresh:
        #save the agent back
        out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
        save_pkl(out_file, agent) 

def resample_time(seconds, 
                  out_pth, 
                  args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #respace the coordinates
    refresh = False
    for tid in tids:
        track = agent.tracks[tid]
        #get the time axis
        t = track['Time'].values.astype(np.int64)/1e9 #to seconds
        total_seconds = t[-1] - t[0]
        #if only one point
        if len(t) == 1:
            continue
        #if multiple points
        else:       
            #if total duration <= new spacing  
            if total_seconds <= seconds:
                continue
            else:
                refresh = True
                #get the new cumsum time
                tq = np.hstack([np.arange(t[0], t[-1], seconds), t[-1:]])               
                #interpolate the other attributes back
                new_track = interpolate_dynamic_data(track, tq, t)
                #update the track dataframe
                agent.tracks[tid] = new_track
    if refresh:
        #save the agent back
        out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
        save_pkl(out_file, agent) 
    
def resample_time_global(time, 
                         out_pth, 
                         args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #respace the coordinates
    for tid in tids:
        track = agent.tracks[tid]
        #get the time axis
        t = track['Time'].values.astype(np.int64)/1e9 #to seconds
        #if there's only 1 ping, just shift the time
        if len(t) == 1:
            new_time_id = np.argmin(np.abs((time - t[0])))
            track.loc[:,'Time'] = pd.to_datetime([time[new_time_id]], 
                                                 unit='s')
        #if theres more than one ping
        else:
            #if track timesteps cross over multiple time gaps   
            resample_times = time[(time>=t[0]) & (time<=t[-1])]
            #if all track pings are contained within one time gap
            if len(resample_times)==0:
                new_time_ids = [np.argmin(np.abs((time - tt))) for tt in t]
                new_time_ids = sorted(np.unique(new_time_ids))
                resample_times = time[new_time_ids]
            #interpolate the other attributes back
            new_track = interpolate_dynamic_data(track, resample_times, t)
            new_track['Time'] = pd.to_datetime(resample_times, unit='s')
            #update the track dataframe
            agent.tracks[tid] = new_track
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 

def compute_coursing(method, 
                     crs, 
                     out_pth,
                     args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, recompute coursing
    for tid in tids:
        track = agent.tracks[tid]
        if len(track) == 1:
            coursing = [np.nan]
        else:
            #recompute coursing based on points
            coursing = points_to_coursing(track['X'].values, 
                                          track['Y'].values, 
                                          crs,
                                          method)
        track.loc[:, 'Coursing'] = coursing  
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 

def compute_turning_rate(method, 
                         out_pth, 
                         args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, recompute coursing
    for tid in tids:
        track = agent.tracks[tid]
        if len(track) == 1:
            turn = [np.nan]
        else:
            coursing = track['Coursing'].values
            dc = np.diff(coursing)
            dt = np.diff(track['Time'].values) / np.timedelta64(1, 's')
            turn = dc/dt
            if method == 'forward':
                turn = np.hstack([turn, turn[-1:]])
            elif method == 'backward':
                turn = np.hstack([turn[:1], turn])
            else:
                turn1 = np.hstack([turn, turn[-1:]])
                turn2 = np.hstack([turn[:1], turn])
                turn = (turn1+turn2)/2
        track.loc[:, 'Turning Rate'] = turn
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 

def compute_speed(method, 
                  out_pth, 
                  args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, recompute coursing
    for tid in tids:
        track = agent.tracks[tid]
        if len(track) == 1:
            speed = [np.nan]
        else:
            coords = track[['X','Y']].values
            dx = np.diff(coords[:,0])
            dy = np.diff(coords[:,1])
            dr = np.sqrt(dx**2+dy**2)
            dt = np.diff(track['Time'].values) / np.timedelta64(1, 's')
            speed = dr/dt
            if method == 'forward':
                speed = np.hstack([speed, speed[-1:]])
            elif method == 'backward':
                speed = np.hstack([speed[:1], speed])
            else:
                speed1 = np.hstack([speed, speed[-1:]])
                speed2 = np.hstack([speed[:1], speed])
                speed = (speed1+speed2)/2
        track.loc[:,'Speed'] = speed
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 

def compute_acceleration(out_pth, 
                         args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, recompute coursing
    for tid in tids:
        track = agent.tracks[tid]
        if len(track) == 1:
            accel = [np.nan]
        else:
            time = track['Time'].values
            speed = track['Speed'].values
            #add acceleration to tdf
            dt = pd.DataFrame(data={0:time,
                                    1:pd.Series(time).shift(1).values,
                                    -1:pd.Series(time).shift(-1).values})
            ds = pd.DataFrame(data={0:speed,
                                    1:pd.Series(speed).shift(1).values,
                                    -1:pd.Series(speed).shift(-1).values})
            accel1 = (ds[0] - ds[-1]) / (dt[0] - dt[-1]).apply(lambda x: x.total_seconds())
            accel2 = (ds[1] - ds[0]) / (dt[1] - dt[0]).apply(lambda x: x.total_seconds())
            accel = (accel1+accel2)/2
        track.loc[:,'Acceleration'] = accel
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 
    
def smooth_corners(refinements, 
                   out_pth, 
                   args):
    #DETECT THE CORNERS BASED ON THRESHOLD??
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, smooth the corners
    refresh = False
    for tid in tids:
        track = agent.tracks[tid]
        #if only 1 point
        if len(track) == 1:
            continue
        #if 2 points and same
        elif len(track) == 2 and start_equals_end(track):
            continue
        #if multiple points
        else:
            refresh = True
            coords = track[['X','Y']].values
            for _ in range(refinements):
                L = coords.repeat(2, axis=0)
                R = np.empty_like(L)
                R[0] = L[0]
                R[2::2] = L[1:-1:2]
                R[1:-1:2] = L[2::2]
                R[-1] = L[-1]
                coords = L * 0.75 + R * 0.25
            #get old and new cumsum distances
            olddist = np.cumsum([[0] + ((np.diff(track['X'])**2 + np.diff(track['Y'])**2)**0.5).tolist()])
            newdist = np.cumsum([[0] + ((np.diff(coords[:,0])**2 + np.diff(coords[:,1])**2)**0.5).tolist()])
            #interpolate the other attributes back
            new_track = interpolate_dynamic_data(track, newdist, olddist)
            #update the track dataframe, overwrite the X/Y
            agent.tracks[tid] = new_track
            new_track.loc[:, 'X'] = coords[:,0]
            new_track.loc[:, 'Y'] = coords[:,1]
    if refresh:
        #save the agent back
        out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
        save_pkl(out_file, agent) 

def start_equals_end(track):
    xstart, ystart = track.iloc[0][['X','Y']].values
    xend, yend = track.iloc[-1][['X','Y']].values
    if xstart == xend and ystart == yend:
        return True
    else:
        return False

def decimate_tracks(epsilon, 
                    out_pth, 
                    args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, smooth the corners
    refresh = False
    for tid in tids:
        track = agent.tracks[tid]
        #if only 1 point
        if len(track) == 1:
            continue
        #if 2 points and same
        elif len(track) == 2 and start_equals_end(track):
            continue
        else:
            refresh = True
            _coords = track[['X','Y']].values
            coords = decimate(_coords, epsilon)
            #get old and new cumsum distances
            olddist = np.cumsum([[0] + ((np.diff(_coords[:,0])**2 + np.diff(_coords[:,1])**2)**0.5).tolist()])
            newdist = np.cumsum([[0] + ((np.diff(coords[:,0])**2 + np.diff(coords[:,1])**2)**0.5).tolist()])
            #interpolate the other attributes back
            new_track = interpolate_dynamic_data(track, newdist, olddist)
            #update the track dataframe, overwrite the X/Y
            agent.tracks[tid] = new_track
            new_track.loc[:, 'X'] = coords[:,0]
            new_track.loc[:, 'Y'] = coords[:,1]
    if refresh:
        #save the agent back
        out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
        save_pkl(out_file, agent) 

def characteristic_tracks(stop_threshold,
                          turn_threshold,
                          min_distance,
                          max_distance,
                          min_stop_duration, 
                          out_pth,
                          inplace,
                          args):
    #split args
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
        if len(track) == 1:
            keeps = [0]
        #if 2 points and same
        elif len(track) == 2: #and start_equals_end(track):
            keeps = [0,1]
        else:
            #characteristic indices
            keeps = characteristic_indices(track['X'],
                                           track['Y'],
                                           track['Time'],
                                           track['Speed'],
                                           track['Coursing'],
                                           stop_threshold,
                                           turn_threshold,
                                           min_distance,
                                           max_distance,
                                           min_stop_duration)
        #add to track dataframe
        track.loc[:, 'Characteristic'] = False
        track.loc[keeps, 'Characteristic'] = True
        #if modifying data in place
        if inplace:
            #if all keepers
            if len(keeps) == len(track):
                continue
            else:
                #update the track dataframe
                agent.tracks[tid] = track.iloc[keeps].reset_index(drop=True)
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 

def simplify_stops(stop_threshold,
                   min_stop_duration, 
                   max_drift_distance,
                   out_pth,
                   args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, smooth the corners
    refresh = False
    for tid in tids:
        track = agent.tracks[tid]
        if len(track) == 1:
            continue
        #if 2 points and same
        elif len(track) == 2 and start_equals_end(track):
            continue
        else:
            keeps = simplified_stop_indices(track['X'],
                                            track['Y'],
                                            track['Time'],
                                            track['Speed'],
                                            stop_threshold,
                                            min_stop_duration,
                                            max_drift_distance)
            #if all keepers
            if len(keeps) == len(track):
                continue
            else:
                refresh = True
                #update the track dataframe
                agent.tracks[tid] = track.iloc[keeps].reset_index(drop=True)
    if refresh:
        #save the agent back
        out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
        save_pkl(out_file, agent) 

def proximity_to_object(shapes, 
                        data_cols,
                        meta_cols,
                        args):
    rows = []
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, get proximity
    for tid in tids:
        track = agent.tracks[tid]
        points1 = track[['X','Y']].values
        #loop over the different shapes
        nearests = []
        min_dists = []
        for points2 in shapes:
            #if both 1 point
            if len(points1)==1 and len(points2)==2:
                nearest_points = {'min_dist': ((points2-points1)**2).sum()**0.5,
                                  'p1': points1[0],
                                  'p2': points2[0],
                                  'idx1': 0,
                                  'idx2': 0,
                                  'frac1': 0,
                                  'frac2': 0}
            #if first is only 1 point
            elif len(points1)==1:
                nearest_points = get_nearest_points(points1, points2)
            #if second is only 1 point
            elif len(points2)==1:
                _np = get_nearest_points(points2, points1)
                new_keys = [k.replace('1','2') if '1' in k 
                            else  k.replace('2','1') 
                            for k in _np.keys()]
                nearest_points = dict(zip(new_keys, _np.values()))
            #both are multiple points
            else:
                _np1 = get_nearest_points(points1, points2)
                _np2 = get_nearest_points(points2, points1)
                if _np2['min_dist'] < _np1['min_dist']:
                    new_keys = [k.replace('1','2') if '1' in k 
                            else  k.replace('2','1') 
                            for k in _np2.keys()]
                    nearest_points = dict(zip(new_keys, _np2.values()))
                else:
                    nearest_points = _np1
                #check for intersection
                _segment1 = points1[nearest_points['idx1']:nearest_points['idx1']+2]
                _segment2 = points2[nearest_points['idx2']:nearest_points['idx2']+2]
                #only if both segments (could be end points of the lines)
                if len(_segment1)>1 and len(_segment2)>1:
                    segment1 = LineString(_segment1)
                    segment2 = LineString(_segment2)
                    #if they intersect
                    if segment1.intersects(segment2):
                        pt = segment1.intersection(segment2)
                        #only handle if point, if line then it'll already be zero
                        if isinstance(pt, Point):
                            #reset the nearest pt info
                            nearest_points['min_dist'] = 0
                            nearest_points['p1'] = np.array(pt.xy).T[0]
                            nearest_points['p2'] = nearest_points['p1']
                            seg1_dist = segment1.length
                            seg2_dist = segment2.length
                            seg1_dt = ((nearest_points['p1'] - _segment1[0])**2).sum()**0.5 / seg1_dist
                            seg2_dt = ((nearest_points['p2'] - _segment2[0])**2).sum()**0.5 / seg2_dist
                            nearest_points['frac1'] = seg1_dt
                            nearest_points['frac2'] = seg2_dt
            #append to nearest list
            nearests.append(nearest_points) 
            min_dists.append(nearest_points['min_dist'])
        #get the closest of all shapes
        idx = np.argmin(min_dists)
        nearest = nearests[idx]
        #interpolate the data to this point
        if 'Time' not in data_cols: data_cols += ['Time']
        sub = track.iloc[nearest['idx1']:nearest['idx1']+2][data_cols]
        #if just a point, data already there
        if len(sub)==1:
            row = sub.iloc[0].to_dict()
        else:
            row = interpolate_dynamic_data(sub, [nearest['frac1']], [0,1]).iloc[0].to_dict()
        row['X'] = nearest['p1'][0]
        row['Y'] = nearest['p1'][1]
        row['geometry'] = Point(nearest['p1'])
        row['Agent ID'] = agent.agent_meta['Agent ID']
        row['Track ID'] = tid
        deltar = ((nearest['p1'] - nearest['p2'])**2).sum()**0.5
        row['Min Distance'] = deltar
        #add meta cols
        for col in meta_cols:
            row[col] = agent.agent_meta[col]
        #append to master rows
        rows.append(row)
    return rows

def get_nearest_points(points1, points2):
    min_dist = 1e9
    min_p1 = (0,0)
    min_p2 = (0,0)
    min_i = 0
    min_j = 0
    frac_i = 0
    frac_j = 0
    updated = False
    for i in range(len(points1)):
        P = points1[i]
        for j in range(len(points2)-1):
            segment = points2[j:j+2]
            dist1 = perpendicular_distance(segment[0], segment[1], P)
            _dist2 = (((segment-P)**2).sum(axis=1)**0.5)
            dist2 = _dist2.min()
            dist2_idx =_dist2.argmin()
            #if closest than current min, and lower than point-point distance
            if dist1 < min_dist and dist1 < dist2:
                x, y = find_intersection_point(segment[0], segment[1], P)
                if (x >= segment[:,0].min() and 
                    x <= segment[:,0].max() and 
                    y >= segment[:,1].min() and
                    y <= segment[:,1].max()): #falls within segment
                    min_dist = dist1
                    min_p1 = P
                    min_p2 = (x,y)
                    min_i = i
                    min_j = j
                    frac_i = 0
                    dist_j = ((x-segment[0][0])**2 + (y-segment[0][1])**2)**0.5
                    frac_j = dist_j / (np.diff(segment, axis=0)**2).sum()**0.5
                    updated = True
                else:
                    updated = False
            #if point-point distance smaller and smaller than current min
            if not updated and dist2 < min_dist:
                min_dist = dist2
                min_p1 = P
                min_p2 = segment[dist2_idx]
                min_i = i
                min_j = j
                frac_i = 0
                frac_j = dist2_idx
            #if current min still lowest
            else:
                pass  
            #reset updater 
            updated = False  
    return {'min_dist': min_dist,
            'p1': min_p1,
            'p2': min_p2,
            'idx1': min_i,
            'idx2': min_j,
            'frac1': frac_i,
            'frac2': frac_j}
            
def find_intersection_point(A, B, P):
    x1, y1 = A
    x2, y2 = B
    x0, y0 = P
    
    if x2 == x1:  # Line AB is vertical
        return (x1, y0)
    if y2 == y1:  # Line AB is horizontal
        return (x0, y1)
    
    m_AB = (y2 - y1) / (x2 - x1)
    m_perpendicular = -1 / m_AB
    
    c_AB = y1 - m_AB * x1
    c_perpendicular = y0 - m_perpendicular * x0
    
    x = (c_perpendicular - c_AB) / (m_AB - m_perpendicular)
    y = m_AB * x + c_AB
    
    return (x, y)

def perpendicular_distance(A, B, P):
    """
    Calculate the perpendicular distance from a point P to a line defined by points A and B.
    
    Parameters:
    A, B: Points defining the line, given as tuples (x, y).
    P: The point from which the distance is measured, given as a tuple (x, y).
    
    Returns:
    The perpendicular distance from P to the line AB.
    """
    x1, y1 = A
    x2, y2 = B
    x0, y0 = P
    
    # Apply the formula
    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
    
    distance = numerator / denominator
    return distance

def proximities(db1, 
                db2, 
                bins, 
                relative,
                data_cols,
                meta_cols,
                i):
    #if bins
    if bins is not None:
        bins = sorted(set(bins))
    #empty output
    rows = []
    #first spatial and temporal intersection pass
    tid = db1.index[i]
    vid = tid.rsplit('_', 1)[0]
    box = db1.iloc[i]['geometry']
    start = db1.iloc[i]['Start Time']
    end = db1.iloc[i]['End Time']            
    # space_ids = np.argwhere(db2.geometry.intersects(box).values).flatten()
    time_ids = np.argwhere((end >= db2['Start Time']).values & (db2['End Time'] >= start).values).flatten()
    #indices in db2 that intersect with track i
    # ids2 = np.array([id for id in space_ids if id in time_ids])
    ids2 = np.array(time_ids)
    #if no matches
    if len(ids2) == 0:
        pass
    #if matches
    else:
        #the original track
        v1 = read_pkl(db1.loc[tid,'File'])
        t1 = v1.tracks[tid.rsplit('_', 1)[1]]
        #track ids from db2 to check
        track_ids2 = db2.index[ids2].tolist()
        for track_id2 in track_ids2:
            #get Agent ID before opening
            vid2 = track_id2.rsplit('_',1)[0]
            #skip all self encounters with itself
            if vid == vid2:
                continue
            #get track
            v2 = read_pkl(db2.loc[track_id2, 'File'])
            t2 = v2.tracks[track_id2.rsplit('_', 1)[1]]
            #filter to common timesteps
            common = t1['Time'].isin(t2['Time'].values)
            common_times = t1['Time'].loc[common]
            common_t1 = t1.loc[common].copy()
            common_t2 = t2[t2['Time'].isin(common_times.values)].copy()
            #calculate distances
            dist = ((common_t1['X'].values - common_t2['X'].values)**2 +
                    (common_t1['Y'].values - common_t2['Y'].values)**2)**0.5
            #if bins
            if bins is not None:
                #if only 1 ping, cant calculate durations
                if len(dist) < 2:
                    continue
                #if more than 1
                elapsed = (common_times - common_times.iloc[0]).dt.total_seconds().values
                if relative:
                    time_spent = {'Time':common_times.tolist(),
                                  **{b:[0]*len(common_times) for b in bins}}
                else:
                    time_spent = dict(zip(bins, [0]*len(bins)))
                durations = np.diff(elapsed)
                #loop over bins
                for bin in bins:
                    #loop over distances/durations
                    for j in range(len(dist)-1):
                        #get segment
                        d1 = dist[j]
                        d2 = dist[j+1]
                        #check if touching or below
                        if d1 <= bin and d2 <= bin:
                            if relative:
                                time_spent[bin][j] += durations[j]
                            else:
                                time_spent[bin] += durations[j]
                        #check if crossing
                        elif (d1<bin and d2>bin) or (d1>bin and d2<bin):
                            if relative:
                                frac = (bin-d1)/(d2-d1)
                                time_spent[bin][j] += durations[j]*frac
                            else:
                                frac = (bin-d1)/(d2-d1)
                                time_spent[bin] += durations[j]*frac
                        #check if above
                        else:
                            pass
            #if relative append the rows
            if relative:
                rows.append(time_spent)
            #if not create non relative rows
            else:
                #get the minimum distance
                idx = np.argmin(dist)
                min_dist = dist[idx]
                #concat the dataframes
                if 'Time' not in data_cols: data_cols += ['Time']
                if 'X' not in data_cols: data_cols += ['X']
                if 'Y' not in data_cols: data_cols += ['Y']
                common_t1 = common_t1[data_cols]
                common_t2 = common_t2[data_cols]
                common_t1.columns = [c+'_0' for c in common_t1.columns]
                common_t2.columns = [c+'_1' for c in common_t2.columns]
                row = pd.concat([common_t1.iloc[idx], 
                                common_t2.iloc[idx]]).to_dict()
                #add meta data cols
                for col in meta_cols:
                    row[f'{col}_0'] = v1.agent_meta[col]
                    row[f'{col}_1'] = v2.agent_meta[col] 
                row['Minimum Distance'] = min_dist
                row['Agent ID_0'] = v1.agent_meta['Agent ID']
                row['Agent ID_1'] = v2.agent_meta['Agent ID']
                row['Track ID_0'] = tid
                row['Track ID_1'] = track_id2
                row['Common Timesteps'] = len(common_times)
                if bins is not None:
                    row.update(time_spent)            
                rows.append(row)
    if relative:
        #it's possible t1 has 1 ping, in which case no rows would have been appended
        #this screws up the grouping in the next step, easier to just make a blank one
        if len(rows) == 0:
            return pd.DataFrame(columns=['Time']+bins)
        else:
            rows = pd.concat([pd.DataFrame(r) for r in rows]).reset_index(drop=True)
            rows = rows.groupby('Time').agg(max).reset_index()
    return rows
 
def _get_closest_encounters(t1, t2, time, distance, filter_min):
    x0, y0 = t1['X'].values, t1['Y'].values
    x1, y1 = t2['X'].values, t2['Y'].values
    time1 = np.array(t1['Time'].tolist())
    time2 = np.array(t2['Time'].tolist())
    #check where the times overlap
    common_start = max([time1[0], time2[0]])
    common_end = min([time1[-1], time2[-1]])
    #make empty output
    encounters = [] #[dist, timediff, id0, id1]
    #if they dont overlap
    if common_start > common_end:
        return encounters
    #if they do overlap
    count, min_idx, min_dist = 0, 0, 999e9
    for i in range(len(x0)):
        #distance to point from track1
        dx = x0[i] - x1
        dy = y0[i] - y1
        dr = dx**2 + dy**2
        #closest point
        idx = np.argmin(dr)
        #dx and dt at this point
        deltar = dr[idx]**0.5
        deltat = np.abs(time2[idx]-time1[i]).total_seconds()
        #if passing thresholds
        if deltar <= distance and deltat <= time:
            #if closer than last closest
            if deltar <= min_dist:
                #now the closest
                min_dist = deltar
                min_idx = count
            #increase encounter counter
            count += 1
            #append the encounter
            encounters.append([deltar,
                               deltat,
                               i,
                               idx])
    #if only minimum
    if filter_min and len(encounters)>0:
        encounters = [encounters[min_idx]]
    return encounters
   
def encounters(db1, 
               db2, 
               distance, 
               time, 
               data_cols, 
               meta_cols,
               filter_min,
               i):
    rows = []
    #first spatial and temporal intersection pass
    tid = db1.index[i]
    box = db1.iloc[i]['geometry']
    start = db1.iloc[i]['Start Time']
    end = db1.iloc[i]['End Time']            
    # space_ids = np.argwhere(db2.geometry.intersects(box).values).flatten()
    time_ids = np.argwhere((end >= db2['Start Time']).values & (db2['End Time'] >= start).values).flatten()
    #indices in db2 that intersect with track i
    # ids2 = np.array([id for id in space_ids if id in time_ids])
    ids2 = np.array(time_ids)
    if len(ids2) == 0:
        pass
    else:
        #the original track
        a1 = read_pkl(db1.loc[tid,'File'])
        t1 = a1.tracks[tid.rsplit('_', 1)[1]]
        #track ids from db2 to check
        track_ids2 = db2.index[ids2].tolist()
        #loop over tracks
        for track_id2 in track_ids2:
            #skip self encounters
            if tid == track_id2:
                continue
            #get track
            a2 = read_pkl(db2.loc[track_id2, 'File'])
            t2 = a2.tracks[track_id2.rsplit('_', 1)[1]]
            #get encounter dist/time
            encounters = _get_closest_encounters(t1, t2, time, distance, filter_min)
            for encounter in encounters:
                deltar, deltat, idx1, idx2 = encounter
                #build the dataframe row
                p1 = Point(t1['X'].iloc[idx1], t1['Y'].iloc[idx1])
                p2 = Point(t2['X'].iloc[idx2], t2['Y'].iloc[idx2])
                center_x = (p1.x + p2.x) / 2
                center_y = (p1.y + p2.y) / 2
                center = Point(center_x, center_y)
                radius = p1.distance(p2) / 2
                circle = center.buffer(radius)
                row = {'Agent ID_0': a1.agent_meta['Agent ID'],
                    'Track ID_0': tid,
                    'Agent ID_1': a2.agent_meta['Agent ID'],
                    'Track ID_1': track_id2,
                    'X_0': t1['X'].iloc[idx1],
                    'Y_0': t1['Y'].iloc[idx1],
                    'X_1': t2['X'].iloc[idx2],
                    'Y_1': t2['Y'].iloc[idx2],                    
                    'Time_0': t1['Time'].iloc[idx1],
                    'Time_1': t2['Time'].iloc[idx2],
                    'Minimum Distance': deltar,
                    'Time Difference': deltat,
                    'geometry':circle}
                #add meta data cols
                for col in meta_cols:
                    row[f'{col}_0'] = a1.agent_meta[col]
                    row[f'{col}_1'] = a2.agent_meta[col] 
                #add dynamic data cols
                for col in data_cols:
                    row[f'{col}_0'] = t1[col].iloc[idx1]
                    row[f'{col}_1'] = t2[col].iloc[idx2]    
                rows.append(row) 
    return rows

def intersections(db1, 
                  db2, 
                  time_threshold, 
                  data_cols,
                  meta_cols,
                  i):
    rows = []
    #do the first filter
    tid = db1.index[i]
    box = db1.iloc[i]['geometry']
    start = db1.iloc[i]['Start Time']
    end = db1.iloc[i]['End Time']            
    space_ids = np.argwhere(db2.geometry.intersects(box).values).flatten()
    time_ids = np.argwhere((end >= db2['Start Time']).values & (db2['End Time'] >= start).values).flatten()
    ids = [id for id in space_ids if id in time_ids]
    #indices for db2 to check for this track
    ids2 = np.array(ids)
    #if none
    if len(ids2) == 0:
        pass
    else:        
        #the original track
        a1 = read_pkl(db1.loc[tid,'File'])
        t1 = a1.tracks[tid.rsplit('_', 1)[1]]
        cumdist1 = ((t1['X'].diff()**2 + t1['Y'].diff()**2)**0.5).values
        cumdist1[0] = 0
        #skip 1 pingers
        if len(t1) < 2:
            return rows
        #if more than 1 ping
        track1 = LineString(zip(t1['X'].values, 
                                t1['Y'].values))
        #first check if the bbox intersects with the track
        touches = db2.iloc[ids2]['geometry'].intersects(track1).values
        ids2 = ids2[touches]
        #track ids from db2 to check
        track_ids2 = db2.index[ids2].tolist()
        #loop over tracks
        for track_id2 in track_ids2:
            #skip self intersections
            if tid == track_id2:
                continue
            #get track
            a2 = read_pkl(db2.loc[track_id2, 'File'])
            t2 = a2.tracks[track_id2.rsplit('_', 1)[1]]
            cumdist2 = ((t2['X'].diff()**2 + t2['Y'].diff()**2)**0.5).values
            cumdist2[0] = 0
            #skip 1 pingers
            if len(t2) < 2:
                continue
            track2 = LineString(zip(t2['X'].values, 
                                    t2['Y'].values))
            #check if it intersects
            if track1.intersects(track2):
                #get the intersections
                cd1, cd2, x, y = find_intersection_points(t1[['X','Y']].values,
                                                          t2[['X','Y']].values).T
                #interpolate dynamic attributes to intersections
                if 'Time' not in data_cols: data_cols += ['Time']
                new_track1 = interpolate_dynamic_data(t1[data_cols], cd1, cumdist1)
                new_track2 = interpolate_dynamic_data(t2[data_cols], cd2, cumdist2)
                new_track1['X'] = x
                new_track1['Y'] = y
                new_track2['X'] = x
                new_track2['Y'] = y
                #reduce to those passing time threshold
                dt = np.abs(new_track1['Time'] - new_track2['Time']).dt.total_seconds()
                mask = dt <= time_threshold
                #if any valid intersections
                if sum(mask) > 0:
                    new_track1 = new_track1.loc[mask]
                    new_track2 = new_track2.loc[mask]
                    #combine into 1 data
                    new_track1['Agent ID'] = a1.agent_meta['Agent ID']
                    new_track2['Agent ID'] = a2.agent_meta['Agent ID']
                    new_track1['Track ID'] = tid
                    new_track2['Track ID'] = track_id2
                    new_track1.columns = [f'{col}_0' for col in new_track1.columns]
                    new_track2.columns = [f'{col}_1' for col in new_track2.columns]
                    _rows = pd.concat([new_track1, new_track2], axis=1)
                    #add meta data cols
                    for col in meta_cols:
                        _rows[f'{col}_0'] = a1.agent_meta[col]
                        _rows[f'{col}_1'] = a2.agent_meta[col] 
                    #add time difference
                    _rows['Time Difference'] = np.abs(_rows['Time_0'] - _rows['Time_1']).dt.total_seconds().values
                    #add geometry column
                    _rows['geometry'] = _rows[['X_0','Y_0']].apply(lambda xy: Point(*xy), axis=1)
                    rows.extend(_rows.to_dict('records'))
                else:
                    pass
    return rows

def find_intersection_points(polyline1, polyline2):
    # Initialize an empty list to store intersection points
    intersection_points = []
    cumsum0 = 0
    # Iterate through line segments of the first polyline
    for i in range(len(polyline1) - 1):
        segment1 = polyline1[i:i + 2]
        # Iterate through line segments of the second polyline
        cumsum1 = 0
        for j in range(len(polyline2) - 1):
            segment2 = polyline2[j:j + 2]
            # Check for intersection between line segments
            x1, y1 = segment1[0]
            x2, y2 = segment1[1]
            x3, y3 = segment2[0]
            x4, y4 = segment2[1]
            # Calculate the determinant
            det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if det != 0:
                # Calculate intersection point coordinates
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / det
                #if valid intersection
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_y = y1 + t * (y2 - y1)
                    di = cumsum0 + (((x2-x1)**2 + (y2-y1)**2)**0.5 * t)
                    dj = cumsum1 + (((x4-x3)**2 + (y4-y3)**2)**0.5 * u)
                    intersection_points.append([di,
                                                dj,
                                                intersection_x, 
                                                intersection_y])
            #increase the cum distance
            cumsum1 += ((x4-x3)**2 + (y4-y3)**2)**0.5
        #increase the cum distance
        cumsum0 += ((x2-x1)**2 + (y2-y1)**2)**0.5         
    # Convert the list of intersection points to a NumPy array
    return np.array(intersection_points)

def is_point_in_bbox(point, bbox):
    (px, py) = point
    (x_min, x_max, y_min, y_max) = bbox
    return x_min <= px <= x_max and y_min <= py <= y_max

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    #calculate determinant
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    #if parallel/coincident
    if div == 0:
       return None
    #if intersection
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    #if within bbox of original segments
    bbox1 = (min(line1[0][0], line1[1][0]), max(line1[0][0], line1[1][0]),
             min(line1[0][1], line1[1][1]), max(line1[0][1], line1[1][1]))
    bbox2 = (min(line2[0][0], line2[1][0]), max(line2[0][0], line2[1][0]),
             min(line2[0][1], line2[1][1]), max(line2[0][1], line2[1][1]))
    if is_point_in_bbox((x,y), bbox1) and is_point_in_bbox((x,y), bbox2):
        return x, y
    else:
        return None

def imprint_geometry(polylines, 
                     out_pth, 
                     args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, smooth the corners
    refresh = False
    for tid in tids:
        track = agent.tracks[tid]
        if len(track) == 1:
            continue
        #if 2 points and same
        elif len(track) == 2 and start_equals_end(track):
            continue
        else:
            xy = track[['X','Y']].values
            old_time = track['Time']
            old_times_cumsum = old_time.diff().dt.total_seconds().cumsum().values
            old_times_cumsum[0] = 0 
            #loop through shapes
            new_rows = []
            for line in polylines:
                #loop over shape segments
                for i in range(len(line)-1):
                    shape_segment = line[i], line[i+1]
                    #loop over track segments
                    for j in range(len(xy)-1):
                        track_segment = xy[j], xy[j+1]
                        #check if intersecting
                        intersection = line_intersection(shape_segment, track_segment)
                        if intersection is not None:
                            #add the new points into the track
                            dxi = (intersection[0] - track_segment[0][0])**2
                            dyi = (intersection[1] - track_segment[0][1])**2
                            dri = (dxi+dyi)**0.5
                            dxt = (track_segment[1][0] - track_segment[0][0])**2
                            dyt = (track_segment[1][1] - track_segment[0][1])**2
                            drt = (dxt+dyt)**0.5
                            frac = dri/drt
                            dtime = (old_time.iloc[j + 1] - old_time.iloc[j]) * frac
                            newtime = old_time.iloc[j] + dtime
                            new_row = {'Time':newtime, 'X':intersection[0], 'Y':intersection[1]}
                            new_rows.append(new_row)
            #if new points to add   
            if len(new_rows) > 0:
                refresh = True
                add_df = pd.DataFrame(new_rows)
                new_times_cumsum = (add_df['Time'] - track['Time'].iloc[0]).dt.total_seconds().values
                add_track = interpolate_dynamic_data(track, new_times_cumsum, old_times_cumsum)
                add_track['X'] = add_df['X'].values
                add_track['Y'] = add_df['Y'].values
                new_track = pd.concat([track, add_track]).sort_values(by='Time').reset_index(drop=True)          
                agent.tracks[tid] = new_track                 
    if refresh:
        #save the agent back
        out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
        save_pkl(out_file, agent)           
    return

def interpolate_raster(interp, 
                       name, 
                       out_pth, 
                       args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, interpolate raster
    for tid in tids:
        track = agent.tracks[tid]
        xy = track[['X','Y']].values
        result = interp(xy)
        track.loc[:, name] = result
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent)    
        
def lateral_distribution(long,
                         xy, 
                         dy,
                         dx, 
                         meta_cols,
                         data_cols,
                         args):
    rows = []
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #make fake polylines
    perpx = -dy
    perpy = dx
    polylines = []
    for center in xy:
        fac = 1e9
        left = center[0]+(perpx*fac) , center[1]+(perpy*fac)
        right = center[0]-(perpx*fac) , center[1]-(perpy*fac)
        segment = np.array([left, right])
        polylines.append(segment)
    #loop over each track, calculate intersections
    for tid in tids:
        track = agent.tracks[tid]
        #skip 1 pingers
        if len(track) < 2:
            continue
        polyline1 = track[['X','Y']].values
        _pl1 = LineString(polyline1)
        #loop over polylines
        intersections = []
        centers = []
        for polyline2,center,lon in zip(polylines, xy, long):
            #only if they intersect
            if LineString(polyline2).intersects(_pl1):
                #get intersection points
                _i = find_intersection_points(polyline1, polyline2)                
                #round for geographic & floating pt errors
                i = np.array(_i).round(8) 
                #reset the longitudinal distance
                i[:,0] = lon
                #fix lateral distances
                #get distance from centreline
                idx = i[:,2] - center[0]
                idy = i[:,3] - center[1]
                dr = (idx**2 + idy**2)**0.5
                #get sign for distance (left or right of centreline)
                pl2_dist = (np.diff(polyline2, axis=0)**2).sum()**0.5
                sign = np.where(i[:,1] <= pl2_dist/2, -1, 1)
                #add signed distance from centreline
                i[:,1] = dr*sign
                #append to list
                intersections.extend(i)
                centers.extend([center]*len(i))
        #if no intersections
        if len(intersections) == 0:
            continue
        #convert to numpy array
        intersections = np.array(intersections)
        centers = np.array(centers)
        #split by direction
        angle = np.degrees(np.arctan2(perpx, perpy)) #TN angle of cross-section from L to R looking down arc
        xd = np.diff(track['X'])
        yd = np.diff(track['Y'])
        dist = np.sqrt(xd ** 2 + yd ** 2)
        oldx = np.cumsum(dist)
        oldx = np.hstack([[0], oldx])
        #prep for interpolation - need coursings for direction checker
        _data_cols = list(set(data_cols+['Coursing']))
        new_track = interpolate_dynamic_data(track[_data_cols], 
                                             intersections[:,0],
                                             oldx)
        angle_diffs = (new_track['Coursing'].values - angle)%360
        direction = np.where(angle_diffs <= 180,  'T', 'F')
        row = pd.DataFrame({'Longitudinal Distance': intersections[:,0],
                            'Lateral Distance': intersections[:,1],
                            'TrackX': intersections[:,2],
                            'TrackY': intersections[:,3],
                            'SliceX': centers[:,0],
                            'SliceY': centers[:,1],
                            'Direction': direction})
        #add data cols
        for col in data_cols:
            row.loc[:,col] = new_track[col]
        #add meta cols
        for col in meta_cols:
            row.loc[:,col] = agent.meta[col]
        row.loc[:, 'Agent ID'] = agent.meta['Agent ID']
        row.loc[:, 'Track ID'] = agent.agent_meta['Track'][tid]['Track ID']
        #append row
        rows.append(row)
    #combine and return
    if len(rows) > 0:
        df = pd.concat(rows)
    else:
        df = pd.DataFrame()
    return df

def time_in_polygon(polygon,
                    edges,
                    meta_cols,
                    data_cols,
                    args):
    rows = []
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, interpolate raster
    for tid in tids:
        track = agent.tracks[tid]
        #skip 1 pingers
        if len(track) < 2:
            continue
        #classify points
        x = track['X'].values
        y = track['Y'].values
        result = np.logical_or(*inpoly2(np.column_stack((x, y)), 
                                        polygon, 
                                        edges))  
        #get segments inside polygon
        result = np.nonzero(result)[0]
        segments = [list(g) for g in consecutive_groups(result)]
        segments = [s for s in segments if len(s) > 1]
        #if none
        if len(segments) == 0:
            continue
        #format data cols for necessary
        if 'X' not in data_cols: data_cols.append('X')
        if 'Y' not in data_cols: data_cols.append('Y')
        if 'Time' not in data_cols: data_cols.append('Time')
        #loop over each segment inside polygon
        for segment in segments:
            #get subtrack inside polygon
            subtrack = track.iloc[segment][data_cols].copy()
            #calculate few columns
            elapsed = (subtrack['Time'].iloc[-1] - subtrack['Time'].iloc[0]).total_seconds()
            xd = np.diff(track['X'])
            yd = np.diff(track['Y'])
            dist = np.sqrt(xd ** 2 + yd ** 2)
            distance = sum(dist)
            #make row
            row = {}
            #add first and last data cols
            first = subtrack.iloc[0].to_dict()
            row.update({k+'_0': v for k,v in first.items()})
            last = subtrack.iloc[-1].to_dict()
            row.update({k+'_1': v for k,v in last.items()})
            #add meta cols
            for col in meta_cols:
                row[col] = agent.meta[col]
            row['Agent ID'] = agent.meta['Agent ID']
            row['Track ID'] = agent.track_meta[tid]['Track ID']
            #add computed
            row['Duration'] = elapsed
            row['Distance Travelled'] = distance
            rows.append(row)
    return rows

def generate_flow_map(characteristic_col,
                      flow_col,
                      args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, route through polygons
    out = []
    for tid in tids:
        track = agent.tracks[tid]
        if characteristic_col is not None:
            track = track[track[characteristic_col]].copy()
        #route through the polygons
        route = [track[flow_col].values[0]]
        current = track[flow_col].values[0]
        for node in track[flow_col].values:
            if node == current:
                pass
            else:
                route.append(node)
                current = node
        #if it never left the same polygon
        if len(route)==1:
            route.append(node)
        #split into dictionary and append to rows
        for i in range(len(route)-1):
            a = route[i]
            b = route[i+1]
            _out = {'Start':a, 
                    'End':b, 
                    'Track ID':agent.track_meta[tid]['Track ID']}
            out.append(_out)
    return out
            
def reduce_to_flow_map(characteristic_col,
                       flow_col,
                       out_pth,
                       points,
                       args):
    #split args
    pkl_files, tracks = args
    #read split agent file
    agent = collect_agent_pkls(pkl_files)
    #reduce to tracks of interest
    if len(tracks) > 0:
        tids = [tid for tid in agent.tracks.keys() 
                if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks]
    else:
        tids = agent.tracks.keys()
    #loop over each track, route through polygons
    for tid in tids:
        track = agent.tracks[tid]
        track = track[track[characteristic_col]].copy()
        track.loc[:,'X'] = track[flow_col].apply(lambda x: points.get(x)[0])
        track.loc[:,'Y'] = track[flow_col].apply(lambda x: points.get(x)[1])
        agent.tracks[tid] = track.reset_index(drop=True)
    #save the agent back
    out_file = f'{out_pth}/{os.path.basename(pkl_files[0])}'
    save_pkl(out_file, agent) 
            
################################################################################