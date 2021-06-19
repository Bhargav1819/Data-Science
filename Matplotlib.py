           
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\bhargav\\OneDrive\\Desktop\\pokemon_alopez247.csv") 
 
           
# =============================================================================
# from scipy.stats import itemfreq
# # Retrieve a list of group types against the
# # number of pokemon(s) belonging to that group
# type_1 = itemfreq(df.iloc[:,2])
# print(type_1)
# # Total number of distinct groups
# type_1_grps = len(type_1)
# print(type_1_grps)
# # Names of group
# type_1_names = type_1[:,0]
# print(type_1_names)
# # Pokemon count particular to each group
# type_1_count = type_1[:,1]
# print(type_1_count)
# 
#            
# type_1_grps = np.arange(type_1_grps)
# bar_width = 0.5
# =============================================================================
# =============================================================================
# plt.bar(type_1_grps, type_1_count, bar_width,
#                  alpha = 0.5,   # tranparency factor
#                  color = 'g',   # color factor
#                  label='Pokemon count respective to their Type_1')
# plt.legend(loc='upper right')
# plt.xticks(type_1_grps + bar_width, type_1_names)
#  
# =============================================================================


      
# =============================================================================
# n_grps = np.arange(5)
# bar_width = 0.4
# men_sc = [20, 30, 10, 50, 90]
# err_men_sc = [2, 3, 4, 5, 4]
# women = [10, 123, 19, 60, 40]
# err_women_sc = [1, 6, 2, 8, 7]
#      
# bar_width=0.4
# plt.bar(n_grps-0.2, men_sc, bar_width, yerr=err_men_sc, label="Men")
# 
# plt.bar(n_grps+bar_width-0.2, women, bar_width, yerr=err_women_sc, label="Women")
# 
# plt.legend(loc='best',shadow='True')
# plt.grid()
# =============================================================================

#plt.xticks(n_grps, list("ABCDE"))


           
# =============================================================================
# import random
# import numpy as np
# # Creating 18 random colors range (0 - 1)
# clrs = np.linspace( 0, 1, 18 )
# random.shuffle(clrs)
# print(clrs)
# # Creating final list of 18 random colors
# colors = []
# for i in range(0,18):
#     idx = np.arange( 0, 18, 1 )
#     random.shuffle(idx)
#     r = clrs[idx[0]]
#     g = clrs[idx[1]]
#     b = clrs[idx[2]]
#     a = clrs[idx[3]]
#     colors.append([r, g, b, a])
#      
# print("The colours are",colors)
# bar_graph = plt.bar(type_1_grps, type_1_count, bar_width,
#                  alpha = 0.5,   # tranparency factor
#                  color = colors)   # color factor
#                   
#            
# plt.legend(bar_graph, 
#            type_1_names,                    # List of group names
#            bbox_to_anchor=(1.128,1.015))   # Position of legend
#             
#            
# plt.xticks(type_1_grps + bar_width/2, type_1_names)
# plt.xlabel('Type')
# plt.ylabel('Pokemon count')
# plt.title('Number of Pokemon per Type_1')
# plt.grid()
# plt.ylim(0,130)
#  
# 
# 
# =============================================================================



           
# =============================================================================
# df_pie = df[['Type_1', 'Attack', 'Defense', 'Speed', 'HP']].copy()
# print(df_pie.head())
#  
#            
# from scipy.stats import itemfreq
# frequent_grp = itemfreq(df_pie.iloc[:,0])
# frequent_grp = np.array(sorted(frequent_grp, key=lambda x: x[1]))[::-1][0:4,:]
# print(frequent_grp)
#  
# print(df_pie.loc[:,'Type_1'].str.contains(r'(Water|Normal|Grass|Bug)'))         
# df_pie = df_pie.loc[df_pie.loc[:,'Type_1'].str.contains(r'(Water|Normal|Grass|Bug)')]
# print(df_pie)
#  
#            
# # Names of the group
# type_1_names = frequent_grp[:,0]
# print(type_1_names)
# # Mean of samples for each feature corresponding to all 4 group 
# df_grp = df_pie.groupby('Type_1').mean()
# print(df_grp)
#  
#            
# names = df_grp.columns
# colors = ['gold', 'lightcoral', 
#               'yellowgreen', 'lightskyblue']
# explode = (0, 0, 0, 0.1)  # takes out only the 4th slice 
# fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2)
# ax = [ax1, ax2, ax3, ax4]
# for i in range(0,4):
#     percent = df_grp.iloc[i,:]
#     ax[i].pie(percent, explode = explode,
#             labels = names, colors = colors,
#             autopct='%.2f%%',   # display value
#             shadow=True,
#             startangle=90)
#     ax[i].set_aspect('equal')
#     ax[i].set_title(type_1_names[i])
# plt.suptitle('Comparing major features of 4 most frequent Pokemon Group',
#              fontsize = 14,
#              fontweight = 'bold')
#               
# =============================================================================


# =============================================================================
# =============================================================================
# total_power = df.iloc[:,4]
# print(total_power)
# catch_rate = df.iloc[:,-2]
# print(catch_rate)
# 
# fig, ax = plt.subplots()
# p = ax.scatter(catch_rate,total_power,c='g')
# ax.grid()
# plt.xlabel('Catch Rate')
# plt.ylabel('Total Power')
# plt.title('Pokemon catch rate vs total power',fontweight='bold')
# plt.legend([p],['Pokemons'])
# 
#            
# import matplotlib.patches as patches
# import matplotlib.transforms as transforms
# trans = transforms.blended_transform_factory(
#     ax.transData,ax.transAxes)
# rect = patches.Rectangle((44,0), width=2, height=5,
#                          transform=trans, color='red',
#                          alpha=0.4)
# ax.add_patch(rect)
#  
#            
# catch_rate_45 = df[df.loc[:,'Catch_Rate'] == 45]
# pow_330 = catch_rate_45[catch_rate_45.loc[:,'Total'] <= 330]
# print("Number of such Pokemons:", len(pow_330))
# # Top 10 adamant Pokets
# print(pow_330.loc[:,'Name'].head(10))
# =============================================================================
 



# =============================================================================


# =============================================================================
# from mpl_toolkits import mplot3d 
# import numpy as np 
# import matplotlib.pyplot as plt 
# fig = plt.figure() 
# ax = plt.axes(projection='3d')
# x, y, z = np.random.rand(3,50)
# c = x + y + z 
# print(c)
# ax.scatter(x, y, z, c=c) 
# ax.set_title('3d Scatter plot') 
# ax.set_xlabel('X Label') 
# ax.set_ylabel('Y Label') 
# ax.set_zlabel('Z Label') 
# plt.show()
#            
# ## rotate the axes and update
# from mpl_toolkits.mplot3d import axes3d
# # consiering you saved your Axes under variable ax
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
#      
# =============================================================================

# =============================================================================
# 
# bulbasaur = list(df.iloc[0,4:11])
# charmander = list(df.iloc[3,4:11])
# squirtle = list(df.iloc[6,4:11])
# pokets = [bulbasaur, charmander, squirtle]
# # Converting height from meters to inches
# # Weight remains in kg
# bul_hw = [df.iloc[0,19]*3.281, df.iloc[0,20]]
# char_hw = [df.iloc[3,19]*3.281, df.iloc[3,20]]
# squ_hw =  [df.iloc[6,19]*3.281, df.iloc[6,20]]
# hw = [bul_hw, char_hw, squ_hw]
# 
#            
# 
# # Line plot
# ax1 = plt.subplot2grid((4,3), (0,0), rowspan = 3, colspan=3)
# ax1.plot(bulbasaur,'g-o')
# ax1.plot(charmander,'r-o')
# ax1.plot(squirtle,'b-o')
# ax1.set_xlim(-0.3,6.3)
#  
# =============================================================================


# continuing from previous code
# =============================================================================
# ax2 = plt.subplot2grid((4,3), (1,0), colspan=1)
# ax3 = plt.subplot2grid((4,3), (2,0), colspan=1)
# ax4 = plt.subplot2grid((4,3), (3,0), colspan=1)
# ax = [ax2, ax3, ax4]
# colors = ['g', 'r', 'b']
# for i in range(0, 3):
#     bp = ax[i].boxplot(pokets[i][1:],patch_artist=True)
#     # Adding colors to edges
#     for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#             plt.setp(bp[element], color='k')
#     # Adding color inside the box
#     for patch in bp['boxes']:
#         patch.set(facecolor=colors[i])
#     ax[i].set_ylim(35,70)  
#               
# ax3.annotate('Median', xy=(1.09, 51), 
#               xytext=(1.2, 60),
#             arrowprops=dict(facecolor='wheat', 
#                             shrink=0.001),)
# =============================================================================
        

# =============================================================================
# x = np.arange(0.0, 5.0, 0.1)
# y = np.cos(2 * np.pi * x) * np.exp(-x)
# 
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# ax.plot(x,y,'k-',x,y,'co')
# ax.set_xlabel("Abscissa")
# ax.set_ylabel("ordinate")
# ax.set_title("Main heading \n Figure Title", fontweight='bold')
# ax.xaxis.label.set_color('green')
# ax.yaxis.label.set_color('green')
# ax.grid()
# ax.annotate('θ=60°',xy=(1,0.4))
# ax.annotate('2nd crest', xy=(2,0.18), 
#               xytext=(2.5,0.3),
#             arrowprops=dict(facecolor='green', 
#                             shrink=0.001))
#         
# 
# =============================================================================
 


























