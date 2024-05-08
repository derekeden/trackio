# ################################################################################

# # from .utils import flatten

# import pandas as pd
# from tqdm import tqdm
# import re
# import os
# import json
# import numpy as np
# import requests as r
# from bs4 import BeautifulSoup as bs
# import multiprocessing as mp
# from functools import partial
# import os

# ###############################################################################

# headers = {
#         "accept": "application/json",
#         "accept-encoding": "gzip, deflate",
#         "user-agent": "Mozilla/5.0",
#         "x-requested-with": "XMLHttpRequest"
#     }

# ###############################################################################

# def empty_info():
#    return {'Name':'',
#         'IMO':np.nan,
#         'MMSI':np.nan,
#         'Type':'',
#         'Length':np.nan,
#         'Width':np.nan}

# def myshiptracking_meta(mmsi=None, imo=None, name=None):
#     #set urls
#     base_url = 'https://www.myshiptracking.com'
#     search_url = "https://www.myshiptracking.com/vessels?name="
#     #check for fields
#     if mmsi is not None:
#         search_url = search_url + f'{mmsi}'
#     elif imo is not None:
#         search_url = search_url + f'{imo}'
#     elif name is not None:
#         search_url = search_url + f'{name.upper()}'
#     else:
#         return print('Must provide mmsi, imo or name...')
#     #search for vessel
#     search = r.get(search_url, headers=headers)
#     #if error
#     try:
#         search.raise_for_status()
#     except r.HTTPError:
#         return empty_info()
#     #get matches
#     soup = bs(search.text, features="html.parser")
#     table = soup.find('table', attrs={'id':'table-filter'})
#     matches = table.find('tbody').find_all('tr')
#     #if no matches
#     if len(matches) == 0:
#         return empty_info()
#     #grab first match
#     else:
#         data_url = base_url + matches[0].find('a').get('href')
#         #get ship page
#         page = r.get(data_url, headers=headers)
#         #if error
#         try:
#             page.raise_for_status()
#         except r.HTTPError:
#             return empty_info()
#         #if no error
#         psoup = bs(page.text, features="html.parser")
#         table = psoup.find('table', attrs={'class':'table table-sm my-2'}).tbody.find_all('tr')
#         #format table
#         data = {tr.find_all('th')[0].text:tr.find_all('td')[0].text for tr in table}
#         #name
#         name = psoup.find('h1', attrs={'class':'mb-0'}).text
#         #imo
#         if data['IMO'] in [None, '', '---']:
#             imo = np.nan
#         else:
#             imo = int(data['IMO'])
#         #mmsi
#         if data['MMSI'] in [None, '', '---']:
#             mmsi = np.nan
#         else:
#             mmsi = int(data['MMSI'])
#         #type
#         typ = psoup.find('h2', attrs={'class':'mb-0 h4 opacity-50'}).text
#         #length
#         if data['Size'] in [None, '', '---']:
#             length = np.nan
#         else:
#             length = float(data['Size'].split('x')[0].strip())
#         #width
#         if data['Size'] in [None, '', '---']:
#             width = np.nan
#         else:
#             width = float(data['Size'].split('x')[1].replace('m','').strip())
#         #construct output
#         out = {'Name':name,
#                 'IMO':imo,
#                 'MMSI':mmsi,
#                 'Type':typ,
#                 'Length':length,
#                 'Width':width}
#         return out

# # def vesselfinder_meta(mmsi=None, imo=None, name=None):
# #     #set urls
# #     search_url = "https://www.vesselfinder.com/vessels?name="
# #     base_url = "https://www.vesselfinder.com"
# #     #check for fields
# #     if mmsi is not None:
# #         search_url = search_url + f'{mmsi}'
# #     elif imo is not None:
# #         search_url = search_url + f'{imo}'
# #     elif name is not None:
# #         search_url = search_url + f'{name.upper()}'
# #     else:
# #         return print('Must provide mmsi, imo or name...')
# #     #search for vessel
# #     search = r.get(search_url, headers=headers)
# #     #if it's an error
# #     try:
# #         search.raise_for_status()
# #     except r.HTTPError:
# #         return {'Name':'',
# #                 'IMO':np.nan,
# #                 'MMSI':np.nan,
# #                 'Type':'',
# #                 'Length':np.nan,
# #                 'Width':np.nan}
# #     #find the matches
# #     soup = bs(search.text, features="html.parser")
# #     matches = soup.find_all('div', attrs={"class": "sli"})
# #     #if no matches
# #     if len(matches) == 0:
# #         return empty_info()
# #     #grab the first match
# #     else:
# #         url = soup.find_all('div', attrs={"class": "sli"})[0].parent.get('href')
# #         #get ship page
# #         data_url = base_url + url
# #         page = r.get(data_url, headers=headers)
# #         #if error
# #         try:
# #             page.raise_for_status()
# #         except r.HTTPError:
# #             return empty_info()
# #         #if no error
# #         psoup = bs(page.text, features="html.parser")
# #         #try to find table
# #         try:
# #             table = psoup.find('h2', text='Vessel Particulars').parent.find_all('tr')
# #         except AttributeError:
# #             return empty_info()
# #         #format table
# #         data = {tr.find_all('td')[0].text:tr.find_all('td')[1].text for tr in table}
# #         #name
# #         if data['Vessel Name'] in [None, '']:
# #             name = ''
# #         else:
# #             name = data['Vessel Name']
# #         #imo
# #         if data['IMO number'] in [None, '']:
# #             imo = np.nan
# #         else:
# #             imo = int(data['IMO number'])
# #         #mmsi
# #         group = re.search('MMSI-(\d+)', url)
# #         if group is not None:
# #             mmsi = int(group.group(1))
# #         else:
# #             mmsi = np.nan
# #         #type
# #         if data['Ship type'] in [None, '']:
# #             typ = np.nan
# #         else:
# #             typ = data['Ship type']
# #         #length
# #         if data['Length Overall (m)'] in [None, '', '-']:
# #             length = np.nan
# #         else:
# #             length = float(data['Length Overall (m)'])
# #         #width
# #         if data['Beam (m)'] in [None, '', '-']:
# #             width = np.nan
# #         else:
# #             width = float(data['Beam (m)'])
# #         #construct output
# #         out = {'Name':name,
# #                'IMO':imo,
# #                'MMSI':mmsi,
# #                'Type':typ,
# #                'Length':length,
# #                'Width':width}
# #         return out

# # def marinetraffic_meta(mmsi=None, imo=None, name=None):
# #     #set urls
# #     search_url = "https://www.marinetraffic.com/en/global_search/search?term="
# #     base_url = 'https://www.marinetraffic.com/vesselDetails/vesselInfo/shipid:'
# #     #check for fields
# #     if mmsi is not None:
# #         search_url = search_url + f'{mmsi}'
# #     elif imo is not None:
# #         search_url = search_url + f'{imo}'
# #     elif name is not None:
# #         search_url = search_url + f'{name.upper()}'
# #     else:
# #         return print('Must provide mmsi, imo or name...')
# #     #search for vessel
# #     search = r.get(search_url, headers=headers)
# #     #if it's an error
# #     try:
# #         search.raise_for_status()
# #     except r.HTTPError:
# #         return empty_info()
# #     #get the results of the search
# #     results = search.json()['results']
# #     #if there's none
# #     if len(results) == 0:
# #         return empty_info()
# #     #get the first result
# #     else:
# #         #get ship id
# #         shipid = results[0]['id']
# #         #set url for ship
# #         base_url = base_url + f'{shipid}'
# #         #get ship page
# #         page = r.get(base_url, headers=headers)
# #         #if error
# #         try:
# #             page.raise_for_status()
# #         except r.HTTPError:
# #             return empty_info()
# #         #if not error
# #         data = page.json()
# #         #name
# #         if data['name'] in [None, '']:
# #             name = data['nameAis']
# #         else:
# #             name = data['name']
# #         #imo
# #         if data['imo'] in [None, '']:
# #             imo = np.nan
# #         else:
# #             imo = int(data['imo'])
# #         #mmsi
# #         if data['mmsi'] in [None, '']:
# #             mmsi = np.nan
# #         else:
# #             mmsi = int(data['mmsi'])
# #         #type
# #         if data['type'] in [None, '']:
# #             typ = np.nan
# #         else:
# #             typ = data['type']
# #         #length
# #         if data['length'] in [None, '']:
# #             length = np.nan
# #         else:
# #             length = float(data['length'])
# #         #width
# #         if data['breadth'] in [None, '']:
# #             width = np.nan
# #         else:
# #             width = float(data['breadth'])
# #         #construct output
# #         out = {'Name':name,
# #                'IMO':imo,
# #                'MMSI':mmsi,
# #                'Type':typ,
# #                'Length':length,
# #                'Width':width}
# #         return out

# # def scrape_vessel_info(search):
# #     #search is a pandas dataframe with MMSI/Name/IMO columns (any or all)
# #     search.columns = [c.lower() for c in search.columns]
# #     if 'mmsi' not in search.columns:
# #         search['mmsi'] == np.nan
# #     if 'name' not in search.columns:
# #         search['name'] = np.nan
# #     if 'imo' not in search.columns:
# #         search['imo'] = np.nan
# #     #initialize results
# #     results = []
# #     #loop over rows of search df
# #     for i in tqdm(list(range(len(search))), desc='Scraping vessel info'):
# #         #get row
# #         row = search.iloc[i]
# #         #first check if mmsi is available for search
# #         if isinstance(row['mmsi'], str) or isinstance(row['mmsi'], int):
# #             #loop over fetchers
# #             for fetcher in (marinetraffic_meta,
# #                             vesselfinder_meta,
# #                             myshiptracking_meta):
# #                 res = fetcher(mmsi=row['mmsi'])
# #                 #if empty, pass so it goes to next fetcher
# #                 if res == empty_info():
# #                     pass
# #                 #if not empty, skip to next iteration
# #                 else:
# #                     break
# #             if res == empty_info():
# #                 pass
# #             else:
# #                 results.append(res)
# #                 continue
# #         #second check for name to search
# #         if isinstance(row['name'], str):
# #             #loop over fetchers
# #             for fetcher in (marinetraffic_meta,
# #                             vesselfinder_meta,
# #                             myshiptracking_meta):
# #                 res = fetcher(name=row['name'])
# #                 #if empty, pass so it goes to next fetcher
# #                 if res == empty_info():
# #                     pass
# #                 #if not empty, skip to next iteration
# #                 else:
# #                     continue
# #             if res == empty_info():
# #                 pass
# #             else:
# #                 results.append(res)
# #                 continue
# #         #third check if imo to search
# #         if isinstance(row['imo'], str) or isinstance(row['imo'], int):
# #             #loop over fetchers
# #             for fetcher in (marinetraffic_meta,
# #                             vesselfinder_meta,
# #                             myshiptracking_meta):
# #                 res = fetcher(imo=row['imo'])
# #                 #if empty, pass so it goes to next fetcher
# #                 if res == empty_info():
# #                     pass
# #                 #if not empty, skip to next iteration
# #                 else:
# #                     continue
# #             if res == empty_info():
# #                 pass
# #             else:
# #                 results.append(res)
# #                 continue
# #         #if missing all this, it cant search, return empty
# #         results.append(empty_info())
# #     out = pd.DataFrame(results)
# #     out['MMSI_original'] = search['mmsi'].values
# #     out['IMO_original'] = search['imo'].values
# #     out['Name_original'] = search['name'].values
# #     out['Type_original'] = search['type'].values
# #     out['Length_original'] = search['length'].values
# #     out['Width_original'] = search['width'].values
# #     out.index = search.index
# #     return out

# ################################################################################


# # ################################################################################
# # # QCing missing data
# # ################################################################################

# # #get mask of vessels to filter out for QC
# # vdb = ds.vdb
# # mask1 = vdb['MMSI'].astype(str).apply(len) != 9 #abnormal mmsi
# # mask2 = vdb['Type'].isin(['Other', 'Unknown']) #other AIS code
# # mask3 = vdb['Length'].isnull() #no length
# # mask4 = vdb['Length'] <= 0 #abnormal length
# # mask5 = vdb['Width'].isnull() #no width
# # mask6 = vdb['Width'] <= 0 #abnormal width

# # #get the chunk of the vessel database, only need 6 columns for this
# # keep_cols = ['Name','MMSI','IMO',
# #              'Type', 'Length', 'Width']
# # search = vdb[(mask1|mask2|mask3|mask4|mask5|mask6)][keep_cols]

# # #scrape missing data from web
# # web_qc = aisio.scrape_vessel_info(search)
# # web_qc.to_csv('qc_VesselInfo.csv')

# # #create random forest classifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier

# # keep_cols = ['Xmin','Xmax','Ymin', 'Ymax',
# #                  'Xstart','Ystart','Xend','Yend', 'Length', 'Effective Distance',
# #                  'Min Speed','Mean Speed', 'Max Speed']

# # data = ds.tracks[ds.tracks['Type']!='Unknown']
# # predict_data = ds.tracks[ds.tracks['Type']=='Unknown'][keep_cols].dropna()

# # X = data[keep_cols]
# # X = X.dropna()
# # Y = data.loc[X.index, 'Type']

# # rest = ds.tracks[ds.tracks['Type']=='Unknown'].loc[predict_data.index]

# # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.4)

# # classifier = RandomForestClassifier(n_estimators=1000, max_samples=20)

# # #train
# # classifier.fit(Xtrain, Ytrain)

# # #test
# # test = classifier.predict(Xtest)

# # #predict
# # predict_data['Type'] = classifier.predict(predict_data)

# # #create neural net for type prediction

# # #get final dataframe for QC
# # df1 = web_qc[['Vessel ID', 'MMSI', 'IMO', 'Length', 'Width', 'Type', 'Name']].copy()
# # df1['AISCode'] = np.nan
# # df2 = rest[['MMSI','IMO','Length','Width','AISCode','Name']].copy()
# # df2['Type'] = predict_data['Type']
# # df2['Vessel ID'] = [i.split('_')[0] for i in df2.index]
# # qc_df = pd.concat([df1, df2]).reset_index(drop=True)

# # #fill missing data in dataframe from the existing database
# # qc_df = aisio.fill_vessel_info(qc_df.drop_duplicates(subset='Agent ID'), ds.agents)

# # #merge qc'd metadata
# # ds = ds.merge_vessel_info(qc_df, ncores=4)

# # #refresh metadata
# # ds = ds.refresh_meta(ncores=4)
