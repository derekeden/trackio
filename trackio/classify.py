################################################################################

import numpy as np
import pandas as pd
from inpoly import inpoly2
from more_itertools import consecutive_groups
from shapely import geometry
from shapely.geometry import Point

from .utils import collect_agent_pkls, first_nonzero, save_pkl

################################################################################


def classify_in_polygon(polygon, edges, code, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        # classify points
        x = track["X"].values
        y = track["Y"].values
        result = np.logical_or(
            *inpoly2(np.column_stack((x, y)), polygon, edges, ftol=1e-8)
        )
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)


def classify_in_polygons(polys, to_codes, name, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        # classify points
        x = track["X"].values
        y = track["Y"].values
        # loop over polys
        codes = {}
        route = {}
        for i in range(len(polys)):
            polycode = polys.iloc[i]["Code"]
            poly, edges = polys.iloc[i]["polys"]
            result = np.logical_or(
                *inpoly2(np.column_stack((x, y)), poly, edges, ftol=1e-8)
            )
            codes[f"Code{polycode}"] = result
            route[f"Code{polycode}"] = result * polycode
        # reduce route to single array
        route = pd.DataFrame(route)
        route = route.apply(first_nonzero, axis=1).values
        track.loc[:, name] = route
        # add codes
        if to_codes:
            for key, val in codes.items():
                track.loc[:, key] = val
    # save the file
    save_pkl(pkl_files[0], agent)


def classify_toward_point(point: Point, code: int, args: tuple):
    # split args
    pkl_files, tracks = args

    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track

    for tid in tids:
        track = agent.tracks[tid]
        start_coordinate = geometry.Point(
            track["X"].iloc[0], track["Y"].iloc[0]
        )
        stop_coordinate = geometry.Point(
            track["X"].iloc[-1], track["Y"].iloc[-1]
        )
        distance_start = start_coordinate.distance(point)
        distance_stop = stop_coordinate.distance(point)
        if distance_start > distance_stop:
            result = True
        else:
            result = False

        track.loc[:, f"Code{code}"] = result

    # save the file
    save_pkl(pkl_files[0], agent)


def classify_speed(speed, higher, code, args):
    if higher:
        method = "higher"
    else:
        method = "lower"
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        # classify points
        if method == "higher":
            result = track["Speed"].values >= speed
        else:
            result = track["Speed"].values <= speed
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)
    return


def classify_turns(rate, code, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        # classify points
        result = np.abs(track["Turning Rate"].values) >= rate
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)
    return


def _classify_speed_inpoly(
    x: np.ndarray[float],
    y: np.ndarray[float],
    speeds: np.ndarray[float],
    poly: np.ndarray,
    edges: tuple,
    speed: float,
    higher: bool,
) -> np.ndarray[bool]:
    """Classify if datapoints lie within a given polygon and its speed is higher or lower than a certain velocity.

    Args:
        x (np.ndarray[float]): X-coordinates of the datapoints.
        y (np.ndarray[float]): Y-coordinates of the datapoints.
        speeds (np.ndarray[float]): The speeds pf the datapoints
        poly (np.ndarray): The polygon coordinates by which to classify.
        edges (tuple): The edges of the polygon.
        speed (float):  The speed by which to classify.
        higher (bool): If True, speed of datapoint needs to be higher than the input speed.

    Returns:
        np.ndarray[bool]: Boolean array with true if classification query is correct and false if not.
    """
    isin = inpoly2(np.column_stack((x, y)), poly, edges, ftol=1e-8)
    if higher:
        return np.logical_and(np.logical_or(*isin), (speeds >= speed))
    else:
        return np.logical_and(np.logical_or(*isin), (speeds <= speed))


def classify_speed_in_polygon(
    speed: float,
    poly: np.ndarray,
    edges: tuple,
    higher: bool,
    code: int,
    args: list,
):
    """Classify if datapoints lie within a given polygon and its speed is higher or lower than a certain velocity.

    Args:
        speed (float): The speed by which to classify.
        poly (np.ndarray): The polygon coordinates by which to classify.
        edges (tuple): The edges of the polygon.
        higher (bool): If True, speed of datapoint needs to be higher than the input speed.
        code (int): The numer of the code column to which the classification bool will be written.
        args (list): Essentially the binary file names and track list if specified.
    """
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        # classify points
        speeds = track["Speed"].values
        x, y = track["X"].values, track["Y"].values
        result = _classify_speed_inpoly(
            x, y, speeds, poly, edges, speed, higher
        )
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)


def _classify_trip(x, y, poly1, edges1, poly2, edges2):
    # one way
    startsin1 = np.logical_or(
        *inpoly2(np.column_stack((x[0], y[0])), poly1, edges1, ftol=1e-8)
    )[0]
    endsin2 = np.logical_or(
        *inpoly2(np.column_stack((x[-1], y[-1])), poly2, edges2, ftol=1e-8)
    )[-1]
    way1 = startsin1 and endsin2
    # other way
    startsin2 = np.logical_or(
        *inpoly2(np.column_stack((x[0], y[0])), poly2, edges2, ftol=1e-8)
    )[0]
    endsin1 = np.logical_or(
        *inpoly2(np.column_stack((x[-1], y[-1])), poly1, edges1, ftol=1e-8)
    )[-1]
    way2 = startsin2 and endsin1
    # either
    if way1 or way2:
        return np.array([True] * len(x))
    else:
        return np.array([False] * len(x))


def classify_trip(poly1, edges1, poly2, edges2, code, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        x = track["X"].values
        y = track["Y"].values
        result = _classify_trip(x, y, poly1, edges1, poly2, edges2)
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)


def _classify_touches(x, y, geom):
    if len(x) == 1:
        return np.array([False] * len(x))
    else:
        ls = geometry.LineString(zip(x, y))
    if geom.intersects(ls):
        return np.array([True] * len(x))
    else:
        return np.array([False] * len(x))


def classify_touching(geom, code, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        x = track["X"].values
        y = track["Y"].values
        result = _classify_touches(x, y, geom)
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)


def _classify_stops(time, speed, stop_threshold, min_stop_duration):
    # find the stops
    stops = np.nonzero(speed <= stop_threshold)[0]
    # group the stops
    stop_ids = [list(g) for g in consecutive_groups(stops)]
    stop_ids = [s for s in stop_ids if len(s) > 1]
    # remove stops less than duration threshold
    stop_ids = [
        s
        for s in stop_ids
        if (time[s[-1]] - time[s[0]]).total_seconds() >= min_stop_duration
    ]
    # flatten the list
    stop_ids = [s for stop in stop_ids for s in stop]
    # make the output array
    out = np.array([False] * len(time))
    out[stop_ids] = True
    return out


def classify_stops(stop_threshold, min_stop_duration, code, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        speed = track["Speed"].values
        time = track["Time"]
        result = _classify_stops(
            time, speed, stop_threshold, min_stop_duration
        )
        track.loc[:, f"Code{code}"] = result
    # save the file
    save_pkl(pkl_files[0], agent)


def classify_custom(values, code, args):
    # split args
    pkl_files, tracks = args
    # read split agent file
    agent = collect_agent_pkls(pkl_files)
    # reduce to tracks of interest
    if len(tracks) > 0:
        tids = [
            tid
            for tid in agent.tracks.keys()
            if f'{agent.agent_meta["Agent ID"]}_{tid}' in tracks
        ]
    else:
        tids = agent.tracks.keys()
    # loop over each track
    for tid in tids:
        track = agent.tracks[tid]
        _tid = f"{agent.agent_meta['Agent ID']}_{tid}"
        track.loc[:, f"Code{code}"] = [values.loc[_tid, "Value"]] * len(track)
    # save the file
    save_pkl(pkl_files[0], agent)


################################################################################
