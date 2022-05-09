from pathlib import Path
import pandas as pd
import os

class DataLoader():
  """
  Returns a tuple that contains all needed DataFrames

  `returns:` (`data`, `components`, `offsets`)
  """
  def __init__(self, data_folder, separator: str =','):
    matchers = ['Cmp', 'Kyu', 'Tou']
    matching = [s for s in os.listdir(data_folder) if any(xs in s for xs in matchers)]
    global_string = os.path.normpath(os.path.join(data_folder, '../global/Components width.csv'))
    self.global_Feeder_Data = pd.read_csv(global_string, sep=';')

    for i in matching:
      skip = -1
      success = False
      while True:
        skip = skip +1

        df = pd.read_csv(str(data_folder) + '/' + i, sep=separator, skiprows=skip, encoding='unicode_escape')
        if 'Component Code' in df.columns:
          break
      
      if 'Cmp' in i:
        self.product_components_data = df
      elif 'Kyu' in i:
        self.product_feeder_data = df
      elif 'Tou' in i:
        self.product_data = df

  def __call__(self, *args: any, **kwds: any) -> tuple:
    neededColumns_Data = ['Component Code', 'X', 'Y', 'Task']
    neededColumns_Components = ['Component Code', 'Placement(Acceleration):X', 'Placement(Acceleration):Y', 'Priority Nozzle No.']
    neededColumns_Feeder = ['Component Code', 'FeedStyle', 'ST No.']
    data = self.product_data[neededColumns_Data]#.rename(columns={'Component Code': 'Code'})
    components_data = self.product_components_data[neededColumns_Components]#.rename(columns={'Component Code': 'index', 'Priority Nozzle No.': 'Nozzle_No'})
    components_feeder_data = self.product_feeder_data[neededColumns_Feeder]#.rename(columns={'Component Code': 'index'})
    data = pd.merge(left=data, left_on='Component Code', right=components_data, right_on='Component Code', how='left')
    components_data = pd.merge(left=components_data, left_on='Component Code', right=components_feeder_data, right_on='Component Code', how='left')
    
    data = data.rename(columns={'Component Code': 'Code'})
    components_data = components_data.rename(columns={'Component Code': 'index', 'Priority Nozzle No.': 'Nozzle_No', 'ST No.': 'ST_No'})
    components_feeder_data = components_feeder_data.rename(columns={'Component Code': 'index'})

    # replace commas with decimal points
    data['X'] = data['X'].replace({',': '.'}, regex=True).astype(float)
    data['Y'] = data['Y'].replace({',': '.'}, regex=True).astype(float)

    # calculate the mean of X and Y acceleration
    components_data['mean_acceleration'] = components_data[['Placement(Acceleration):X', 'Placement(Acceleration):Y']].mean(axis=1)
    components_data['mean_acceleration'] = components_data['mean_acceleration']#fillna(1000.0)
    # devide coordinates
    if data['X'].max() > 1000:
      data['X'] = data['X'] / 1000
      data['Y'] = data['Y'] / 1000

    # split offset and drop duplicates
    offsets = data.loc[data['Task'] == 'Repeat Offset']
    zero_offset = pd.DataFrame({'Code': '', 'X': 0, 'Y': 0, 'Task': 'Repeat Offset'}, index=[0])
    offsets = pd.concat([zero_offset, offsets], axis=0)
    offsets = offsets.drop_duplicates()
    offsets = offsets.reindex()
    offsetlist = []
    for index, row in offsets.iterrows():
      offset = (row.X, row.Y)
      offsetlist.append(offset)



    data = data.loc[data['Task'] != 'B Mark Positive Logic']
    
    # drop NaN in data
    data = data.dropna()
    fid = data[(data.Task == 'Fiducial')]
    fid = fid.rename(columns={'Code': 'index'})
    #components = pd.concat([fid,components], axis=0)
    #print(data)

    # create component dataset
    occ = data['Code'].value_counts()
    components = pd.DataFrame(occ, columns=['Code']).reset_index()
    # create pickup coordinates
    components['Pickup_Y'] = 0
    components['Pickup_X'] = range(len(components.index))
    components = pd.merge(left=components, right= components_data, left_on='index', right_on='index', how='left').drop_duplicates()
    #components = pd.merge(left=components, right= components_feeder_data, left_on='index', right_on='index', how='left').drop_duplicates()
    components = pd.merge(left=components, right= self.global_Feeder_Data, left_on='index', right_on='Component Code', how='left').drop_duplicates()
    components = components.drop(['Component Code'], axis=1)
    components['mean_acceleration'] = components['mean_acceleration'].fillna(1000.0)

    self.data = data
    self.components = components
    self.offsets = offsetlist
    return (data, components, offsetlist)

if __name__ == '__main__':
  path =  Path(os.getcwd() + os.path.normpath('/data/24aarab'))
  dataloader = DataLoader(path, separator=',')
  data = dataloader()
  FL = data[1]#.loc[data[1]['FeedStyle'] == 'ST-FL']
  FL#.duplicated()