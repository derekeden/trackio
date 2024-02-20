################################################################################

from .classes.Dataset import Dataset
from .classes.Agent import Agent
from .tools.io import rasterize, create_blank_raster
from .tools.utils import read_pkl as read_agent
from .tools.utils import save_pkl as write_agent
from .tools.processing import clip_to_polygon, clip_to_box
from .tools.mappers import _mappers as mappers
from .tools.mappers import make_col_mapper, make_raw_data_mapper

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