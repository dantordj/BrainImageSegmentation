# naive approach

import pandas as pd
import os

mni_size = 7109137
df = pd.DataFrame()
num_atlasses = len(atlases)
C = np.zeros((mni_size, num_atlasses))
for index, atlas in atlases.items():
  df[index] = atlas.label_vector()
  df[index] = df[index].astype(int)


print(df.head())

order_index = {index: i for i,index in enumerate(atlases.keys())}
print(order_index)
count = 0


for i in conflict_voxels:
  print()
  print(" ---- ")
  print()
  labels = df.loc[i]
  print(labels)
  hist = np.bincount(labels)
  print(hist)


  for index, atlas in atlases.items():
    index_matrix = order_index[index]

    C[i, index_matrix] = hist[labels[index]] * 1. / num_atlasses

  if count % 20000 == 0:
    print(count)
  count += 1
