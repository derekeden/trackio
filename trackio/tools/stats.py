# ################################################################################

# import pandas as pd
# from collections import Counter
# import numpy as np

# ################################################################################

# def vessel_width_stats(dataset, agents=None):
#     if agents is None:
#         agents = dataset.agents
#     else:
#         pass
#     beams = agents[['Type','Width']]
#     beams = beams[beams['Width'] > 0]
#     beams = beams[~beams['Width'].isnull()]
#     _beams = beams.copy()
#     _beams['Type'] = 'All'
#     beams = pd.concat([beams,_beams])
#     stats = beams.groupby('Type')['Width'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats
   
# def vessel_length_stats(dataset, agents=None):
#     if agents is None:
#         agents = dataset.agents
#     else:
#         pass
#     loas = agents[['Type','Length']]
#     loas = loas[loas['Length'] > 0]
#     loas = loas[~loas['Length'].isnull()]
#     _loas = loas.copy()
#     _loas['Type'] = 'All'
#     loas = pd.concat([loas,_loas])
#     stats = loas.groupby('Type')['Length'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats

# def vessel_counts(dataset, agents=None):
#     if agents is None:
#         agents = dataset.agents
#     else:
#         pass
#     counts = agents['Type'].value_counts()
#     counts.loc['All'] = counts.sum()
#     return counts

# def track_counts(dataset, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     counts = tracks['Type'].value_counts()
#     counts.loc['All'] = counts.sum()
#     return counts

# def track_length_stats(dataset, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     ls = tracks[['Type','Track Length']]
#     # ls = ls[ls['Track Length'] > 0]
#     # ls = ls[~ls['Track Length'].isnull()]
#     _ls = ls.copy()
#     _ls['Type'] = 'All'
#     ls = pd.concat([ls,_ls])
#     stats = ls.groupby('Type')['Track Length'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats

# def track_duration_stats(dataset, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     ds = tracks[['Type','Duration']]
#     # ds = ds[ds['Duration'] > 0]
#     # ds = ds[~ds['Duration'].isnull()]
#     _ds = ds.copy()
#     _ds['Type'] = 'All'
#     ds = pd.concat([ds,_ds])
#     stats = ds.groupby('Type')['Duration'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats 

# def npings_per_vessel_stats(dataset, agents=None):
#     if agents is None:
#         agents = dataset.agents
#     else:
#         pass
#     p = agents[['Type','npings']]
#     # p = p[p['npings'] > 0]
#     # p = p[~p['npings'].isnull()]
#     _p = p.copy()
#     _p['Type'] = 'All'
#     p = pd.concat([p,_p])
#     stats = p.groupby('Type')['npings'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats

# def npings_per_track_stats(dataset, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     p = tracks[['Type','npings']]
#     # p = p[p['npings'] > 0]
#     # p = p[~p['npings'].isnull()]
#     _p = p.copy()
#     _p['Type'] = 'All'
#     p = pd.concat([p,_p])
#     stats = p.groupby('Type')['npings'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats

# def ntracks_per_vessel_stats(dataset, agents=None):
#     if agents is None:
#         agents = dataset.agents
#     else:
#         pass
#     p = agents[['Type','ntracks']]
#     # p = p[p['ntracks'] > 0]
#     # p = p[~p['ntracks'].isnull()]
#     _p = p.copy()
#     _p['Type'] = 'All'
#     p = pd.concat([p,_p])
#     stats = p.groupby('Type')['ntracks'].agg(list).apply(lambda x: pd.Series(x).describe())
#     return stats

# def vessels_by_month(dataset, combine=False, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     months = tracks[['Type','Start Time', 'Agent ID']].copy()
#     months['Start Time'] = pd.to_datetime(months['Start Time'])
#     if combine:
#         grouper = months['Start Time'].dt.month
#     else:
#         grouper = months['Start Time'].dt.year.astype(str) + '-' + months['Start Time'].dt.month.astype(str)
#     stats = months.groupby(grouper)[['Type','Agent ID']].agg(list)
#     stats['idx'] = stats['Agent ID'].apply(lambda x: np.unique(x, return_index=True)[1])
#     for i in stats.index:
#             ids = stats.loc[i, 'idx']
#             stats.at[i, 'Type'] = np.array(stats.loc[i, 'Type'])[ids]
#     stats['Type'] = stats['Type'].apply(Counter)
#     out = []
#     for i in range(len(stats)):
#         row = stats.iloc[i]
#         count = 0
#         for k,v in row.Type.items():
#             tmp = {}
#             tmp['Month'] = stats.index[i]
#             tmp['Type'] = k
#             tmp['Count'] = v
#             out.append(tmp)
#             count += v
#         out.append({'Month':stats.index[i],
#                     'Type':'All',
#                     'Count':count})
#     df = pd.DataFrame(out)
#     df['sorter'] = pd.to_datetime(df['Month'], format='%Y-%m')
#     df = df.sort_values(by='sorter').drop(columns='sorter')
#     return df

# def vessels_by_year(dataset, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     years = tracks[['Type','Start Time', 'Agent ID']].copy()
#     years['Start Time'] = pd.to_datetime(years['Start Time'])
#     grouper = years['Start Time'].dt.year
#     stats = years.groupby(grouper)[['Type','Agent ID']].agg(list)
#     stats['idx'] = stats['Agent ID'].apply(lambda x: np.unique(x, return_index=True)[1])
#     for i in stats.index:
#         ids = stats.loc[i, 'idx']
#         stats.at[i, 'Type'] = np.array(stats.loc[i, 'Type'])[ids]
#     stats['Type'] = stats['Type'].apply(Counter)
#     out = []
#     for i in range(len(stats)):
#         row = stats.iloc[i]
#         count = 0
#         for k,v in row.Type.items():
#             tmp = {}
#             tmp['Year'] = stats.index[i]
#             tmp['Type'] = k
#             tmp['Count'] = v
#             out.append(tmp)
#             count += v
#         out.append({'Year':stats.index[i],
#                     'Type':'All',
#                     'Count':count})
#     df = pd.DataFrame(out)
#     df = df.sort_values(by=['Year','Type']).reset_index(drop=True)
#     return df

# def tracks_by_month(dataset, combine=False, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     months = tracks[['Type','Start Time']].copy()
#     months['Start Time'] = pd.to_datetime(months['Start Time'])
#     if combine:
#         grouper = months['Start Time'].dt.month
#     else:
#         grouper = months['Start Time'].dt.year.astype(str) + '-' + months['Start Time'].dt.month.astype(str)
#     stats = months.groupby(grouper)['Type'].agg(Counter)
#     out = []
#     for i in range(len(stats)):
#         row = stats.iloc[i]
#         count = 0
#         for k,v in row.items():
#             tmp = {}
#             tmp['Month'] = stats.index[i]
#             tmp['Type'] = k
#             tmp['Count'] = v
#             out.append(tmp)
#             count += v
#         out.append({'Month':stats.index[i],
#                     'Type':'All',
#                     'Count':count})
#     df = pd.DataFrame(out)
#     return df

# def tracks_by_year(dataset, tracks=None):
#     if tracks is None:
#         tracks = dataset.tracks
#     else:
#         pass
#     years = tracks[['Type','Start Time']].copy()
#     years['Start Time'] = pd.to_datetime(years['Start Time'])
#     grouper = years['Start Time'].dt.year
#     stats = years.groupby(grouper)['Type'].agg(Counter)
#     out = []
#     for i in range(len(stats)):
#         row = stats.iloc[i]
#         count = 0
#         for k,v in row.items():
#             tmp = {}
#             tmp['Year'] = stats.index[i]
#             tmp['Type'] = k
#             tmp['Count'] = v
#             out.append(tmp)
#             count += v
#         out.append({'Year':stats.index[i],
#                     'Type':'All',
#                     'Count':count})
#     df = pd.DataFrame(out)
#     return df

# def vessel_width_dist(dataset, bins=10, density=True):
#     beams = dataset.agents[['Type','Width']]
#     beams = beams[beams['Width'] > 0]
#     beams = beams[~beams['Width'].isnull()]
#     _beams = beams.copy()
#     _beams['Type'] = 'All'
#     beams = pd.concat([beams,_beams])
#     grouped = beams.groupby('Type')['Width'].agg(list)
#     x = grouped.apply(lambda x: np.histogram(x, bins=bins, density=False))
#     out = pd.DataFrame({'Type':x.index, 
#                         'PDF':x.apply(lambda x: x[0]),
#                         'Edges':x.apply(lambda x: x[1])})
#     if density:
#         tot = out.loc['All','PDF'].sum()
#         out['PDF'] = out['PDF'].apply(lambda x: [y/tot for y in x])
#     else:
#         pass
#     return out

# def vessel_length_dist(dataset, bins=10, density=True):
#     loas = dataset.agents[['Type','Length']]
#     loas = loas[loas['Length'] > 0]
#     loas = loas[~loas['Length'].isnull()]
#     _loas = loas.copy()
#     _loas['Type'] = 'All'
#     loas = pd.concat([loas,_loas])
#     grouped = loas.groupby('Type')['Length'].agg(list)
#     x = grouped.apply(lambda x: np.histogram(x, bins=bins, density=False))
#     out = pd.DataFrame({'Type':x.index, 
#                         'PDF':x.apply(lambda x: x[0]),
#                         'Edges':x.apply(lambda x: x[1])})
#     if density:
#         tot = out.loc['All','PDF'].sum()
#         out['PDF'] = out['PDF'].apply(lambda x: [y/tot for y in x])
#     else:
#         pass
#     return out
        
# def track_length_dist(dataset, bins=10, density=True):
#     ls = dataset.tracks[['Type','Track Length']]
#     # ls = ls[ls['Track Length'] > 0]
#     # ls = ls[~ls['Track Length'].isnull()]
#     _ls = ls.copy()
#     _ls['Type'] = 'All'
#     ls = pd.concat([ls,_ls])
#     grouped = ls.groupby('Type')['Track Length'].agg(list)
#     x = grouped.apply(lambda x: np.histogram(x, bins=bins, density=False))
#     out = pd.DataFrame({'Type':x.index, 
#                         'PDF':x.apply(lambda x: x[0]),
#                         'Edges':x.apply(lambda x: x[1])})
#     if density:
#         tot = out.loc['All','PDF'].sum()
#         out['PDF'] = out['PDF'].apply(lambda x: [y/tot for y in x])
#     else:
#         pass
#     return out  

# def track_duration_dist(dataset, bins=10, density=True):
#     ds = dataset.tracks[['Type','Duration']]
#     # ds = ds[ds['Duration'] > 0]
#     # ds = ds[~ds['Duration'].isnull()]
#     _ds = ds.copy()
#     _ds['Type'] = 'All'
#     ds = pd.concat([ds,_ds])
#     grouped = ds.groupby('Type')['Duration'].agg(list)
#     x = grouped.apply(lambda x: np.histogram(x, bins=bins, density=False))
#     out = pd.DataFrame({'Type':x.index, 
#                         'PDF':x.apply(lambda x: x[0]),
#                         'Edges':x.apply(lambda x: x[1])})
#     if density:
#         tot = out.loc['All','PDF'].sum()
#         out['PDF'] = out['PDF'].apply(lambda x: [y/tot for y in x])
#     else:
#         pass
#     return out  

# ################################################################################