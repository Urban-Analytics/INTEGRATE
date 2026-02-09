import os
from pathlib import Path

data_dir =   os.path.join("../../../../data/embeddings/")
lsoas_file = os.path.join("../../../../data/SpatialData/", "LSOAs_2021", "LSOA_2021_EW_BSC_V4.shp")
imd_file = (Path(data_dir) / ".." / "imd" / "File_2_-_IoD2025_Domains_of_Deprivation.xlsx").resolve()

h5_filename = data_dir + "sample_points_cache/street_data.h5"