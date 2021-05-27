import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt

# =============================================================================
# df = pd.DataFrame({'A': [1,2],
#                    'B': "Bhargav",
#                    'C': [2.5,5.6],
#                    'D': 3+4j},index=['R1','R2'])
# print(df)
# print(df.dtypes)
# print(df.info())
# print(df.index[1])
# print(df.columns[3])
# print(df.values[1])
# =============================================================================


# =============================================================================
# df = pd.DataFrame([[0.23,'f1'],[5.36,'f2']],
#                   index = list('pq'),
#                     columns = list('ab'))
# 
# df = df.rename(columns={'a':'A'})
# df = df.rename(index={'p':'P'})
# new = list(np.random.randint(0,2,2))
# df['c']=new
# df['A']=df['A'].astype('complex')
# print(df)
# df['A']=df['A'].astype('complex')
# lst = ['f30','f50','f2','f0'] 
# print(df[df.isin(lst).any(axis=1)])
# =============================================================================



# =============================================================================
# df = pd.DataFrame([[11, 202],
#                     [33, 44]],
#                     index = list('AB'),
#                     columns = list('CD'))
# =============================================================================

#df.to_excel("C:\\Users\\bhargav\\Desktop\\sample.xlsx", sheet_name = 'Sheet1')
#print(df)
# =============================================================================
# a=pd.read_excel('C:\\Users\\bhargav\\Desktop\\sample.xlsx', 'Sheet1')
# print(a)
# =============================================================================

# =============================================================================
# df = pd.read_table("C:\\Users\\bhargav\\Desktop\\sada_chat.txt")
# print(df.values)
# =============================================================================


# =============================================================================
# df = pd.DataFrame([[18,10,5,11,-2],
#                     [2,-2,9,-11,3],
#                     [-4,6,-19,2,1],
#                     [3,-14,1,-2,8],
#                     [-2,2,4,6,13]],
#                   index = list('pqrst'),
#                     columns = list('abcde'))
# 
# 
# new_df = df[df.apply(sum,axis=1)%2==0]
# writer = pd.ExcelWriter("C:\\Users\\bhargav\\Desktop\\file_df.xlsx", engine='xlsxwriter')
# new_df.to_excel(writer, sheet_name = 'Sheet1')
# 
# 
# 
# df_temp=new_df.copy()
# df_temp['m']=df_temp.apply(np.prod, axis = 1)
# df_temp.to_excel(writer, sheet_name = 'Sheet2')
# writer.save()
# 
# print(new_df)
# =============================================================================
# =============================================================================
# 
# df = pd.DataFrame([[15, 12],
#                     [33, 54],
#                     [10, 32]], 
#                     index = list('ABC'),
#                     columns = list('DE'))
# 
# print(len(df))
# =============================================================================
# =============================================================================
# print(df.loc[['A','C'],:])
# print(df.loc[:,'E'])
# print(df.loc['B', 'D'])
# 
# print(df.iloc[:,1])
# =============================================================================

df = pd.read_table("F:\\Chrome Downloads\\chat.txt")

              
''' Generating a boolean sequence which results True if it contains Timestamp '''
bools = df.iloc[:len(df),0].str.contains(r'^\d+\/\d+\/\d+, \d+:\d+:\d+ .M:')
''' Concatenating '''
i = len(df) - 1
while (i >= 0):
    if bools[i] == False:
        df.iloc[i - 1, 0] += ' ' + df.iloc[i, 0]
    i -= 1
''' Dropping rows whose data is appended to source row '''
df = df[bools]  
''' Reformatting index '''
df = df.reset_index(drop = True)    


#   26/08/17, 12:28:58 PM: ‎Messages to this group are now secured with end-to-end encryption.
# 0  26/08/17, 12:36:23 PM: Friend1: *Richard M. St...                                        
# 1  26/08/17, 12:36:37 PM: ?+91 12345 45555: <‎ima...                                        
# 2  26/08/17, 12:36:54 PM: +91 12345 45555: <‎imag...                                        
# 3             26/08/17, 12:37:08 PM: Friend3: ????????????                                        
# 4  26/08/17, 12:37:33 PM: +91 12345 45555: <‎GIF ...                                        
              

df = df.iloc[:,0].str.split('M:', 1, expand=True)
ts = df.iloc[:,0].copy()
ts = ts.reset_index(drop = True)
ts.columns = ['Timestamp']
df = df.iloc[:,1].str.split(':', 1, expand=True)
df = df.reset_index(drop = True)
df.columns = ['Name','Convo'] 


df = df[df['Name'].str.contains(r'\d{2} \d{5} \d{5}') == False] 
              
df = df[df['Convo'].str.contains(r'image omitted') == False]
df = df[df['Convo'].str.contains(r'video omitted') == False]
df = df[df['Convo'].str.contains(r'GIF omitted') == False]
df = df.reset_index(drop = True) 
              

all_chat_list = []

for i in range(len(df['Name'].drop_duplicates())):
    
    temp = df['Convo'][df['Name'] == df['Name'].drop_duplicates().reset_index(drop = True)[i]]
    temp = temp.reset_index(drop = True)
    for j in range(1,len(temp)):
        temp[0] += ' ' + temp[j]
    all_chat_list.append(temp[0])
    del temp 
              
from scipy.stats import itemfreq
''' Friend1 Top 3 words '''
fg = itemfreq(list(all_chat_list)[0].split(' '))
fg = fg[fg[:,1].astype(float).argsort()][::-1]
#print(fg[1:4])
# [['abroad' '4']
# ['the' '3']
#  ['home' '3']]
''' Friend3 Top 3 words '''
fg = itemfreq(list(all_chat_list)[2].split(' '))
fg = fg[fg[:,1].astype(float).argsort()][::-1]
#print(fg[1:4])
# [['company' '3']
#  ["it's" '2']
#  ['hmm' '1']]

# =============================================================================
# ''' Check which row has missing value '''
# print(ts.isnull())
# # 0      False
# # 1      False
# # 2      False
# # 3      False
# # 4      False
# # 5      False
# # .       .
# # .       .
# ''' Check total number of non-missing values '''
# print(ts.count())
# =============================================================================
# 122 


import datetime
splitted_ts= ts.str.split(', ')
for i in range(len(ts)):
    splitted_ts[i][1] += 'M'
    temp = datetime.datetime.strptime(splitted_ts[i][1], '%I:%M:%S %p')
    splitted_ts[i][1] = datetime.datetime.strftime(temp, '%H')

              
hrs = [ splitted_ts[i][1] for i in range(len(splitted_ts)) ]
hrfreq = itemfreq(hrs)
occ = [float(hrfreq[i][1]) for i in range(len(hrfreq))]
hr = [float(hrfreq[i][0]) for i in range(len(hrfreq))]
plt.plot(hr, occ)
plt.grid('on')
plt.xlabel('24 Hours')
plt.ylabel('Frequency')
plt.title('Frequent chat timings') 
              


import numpy as np
import pandas as pd

df = pd.read_csv("F:\\Chrome Downloads\\bridge.csv",header=None)
df.columns=['IDENTIF','RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 
  'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']

df.replace('?', np.NaN, inplace=True) 
df.isna().sum()
dfc = df.columns[df.columns.str.contains("[NHS]$")] 
df1 = df[dfc].dropna(axis=1, thresh=100) 
df2 = df.drop(columns=dfc)
df2[df1.columns] = df1[df1.columns] 
#df2.info() 
df2.dropna(thresh=2,inplace=True)
df2.reset_index(drop=True,inplace=True)
cols = df2.columns[df2.isnull().any()]
df2[cols]=df2[cols].fillna(df2.mode().iloc[0]) 
print(df2.isnull().sum())


# =============================================================================
# 
# from sklearn.impute import SimpleImputer
# SI = SimpleImputer(strategy="most_frequent", copy=False) 
# SI.fit_transform(df2) 
# df2.info()
# 
# =============================================================================




import pandas as pd
sys = ['s1','s1','s1','s1',
        's2','s2','s2','s2']
net_day = ['d1','d1','d2','d2',
        'd1','d1','d2','d2']
spd = [1.3, 11.4, 5.6, 12.3, 
        6.2, 1.1, 20.0, 8.8]
df = pd.DataFrame({'set_name':sys,
                    'spd_per_day':net_day,
                    'speed':spd})
print(df)              
new_df = df.groupby(["set_name","spd_per_day"]).median() 
new_df.columns = pd.MultiIndex.from_arrays([["speed"], ["median"]]) 
new_df=new_df.sort_values(by=("speed","median"))
print(new_df)










































