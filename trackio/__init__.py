################################################################################

from .Dataset import Dataset
from .Agent import Agent
from .io import rasterize, create_blank_raster
from .utils import read_pkl as read_agent
from .utils import save_pkl as write_agent
from .process import clip_to_polygon, clip_to_box
from .maps import _mappers as mappers
from .maps import make_col_mapper, make_raw_data_mapper

################################################################################

def read(raw_files=None, 
         data_files=None,
         data_path='./data'):
    return Dataset(raw_files=raw_files,
                   data_files=data_files,
                   data_path=data_path)
    
def from_df(raw_df,
            data_path='.'):
     return Dataset(raw_df=raw_df,
                    data_path=data_path)
     
def from_gdf(raw_gdf,
             data_path='.'):
     return Dataset(raw_gdf=raw_gdf,
                    data_path=data_path)