# MACS30000: Problem Set #3
# Dr. Evans
#
# Name : Alice Mee Seon Chung 
#       (alice.chung@uchicago.edu)
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import dates 
from datetime import datetime, timedelta
from matplotlib.ticker import MultipleLocator
np.seterr(divide='ignore', invalid='ignore')
...
graph = True
...
if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    year_vec    = (lf_years,) vector, years from beg_year to
                  beg_year + lf_years
    individual  = integer in [0, numdraws-1], index of particular series
                  to plot
    --------------------------------------------------------------------
    '''
    # create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

'''
------------------------------------------------------------------------
Problem 1. A lifetime of emperatures
------------------------------------------------------------------------
'''
# make plot figure 
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)

# file lists
files = ['Indianapolis.csv','Pittsburgh.csv', 'Miami.csv','Washington.csv','Chicago.csv']

# make date to type of datetime type
def day_of_year(date_string):
  return datetime.strptime(date_string, '%Y%m%d').timetuple().tm_yday

# Plot Chicago, Indianapolis, Pittsburgh data
Chicago = False
for file in files:
    if file == 'Chicago.csv':
        Chicago = True
    if file == 'Indianapolis.csv':
        highlight_born = True
    if file == 'Pittsburgh.csv':
        highlight_win = True
    # dataframe of all 5 cities   
    temperture_df = pd.read_csv(file).groupby('DATE').mean().round(2).reset_index()

    # generate the dataframe in form of 366 days
    # the day 1 be September 21 and 366 be September 20
    x_axis=[]
    for date in temperture_df['DATE']:
        if date % 10000 < int(921):
            if (date // 10000) % 4 == 0: # find the leap year to find Feburary 29th
                x_axis.append(day_of_year(str(date)) - day_of_year('20160101')
                              + day_of_year('20161231') - day_of_year('20160921') + 1)
            else:
                x_axis.append(day_of_year(str(date)) - day_of_year('20160101') 
                              + day_of_year('20161231') - day_of_year('20160921') + 2)
        elif (date // 10000) % 4 == 0: # find the leap year to find Feburary 29th
            x_axis.append(day_of_year(str(date)) - (day_of_year('20160921')))
        else:
            x_axis.append(day_of_year(str(date)) - (day_of_year('20160921')) + 1)

    # fill maroon color to highlight during Ricardo's Illinois period
    if Chicago:
        plt.scatter(x_axis, temperture_df['TMAX'],s=0.1 ,marker="o", color='maroon')
        plt.scatter(x_axis, temperture_df['TMIN'],s=0.1, color='maroon',marker="o")
        Chicago = False
    else:
         plt.scatter(x_axis, temperture_df['TMAX'],s=0.1, color='k',marker="o")
         plt.scatter(x_axis, temperture_df['TMIN'],s=0.1, color='k',marker="o")

# to do annotation, set the dataframe again 
born_set = pd.read_csv('Indianapolis.csv').groupby('DATE').mean().round(2).reset_index()
little_league_win_set = pd.read_csv('Pittsburgh.csv').groupby('DATE').mean().round(2).reset_index() 

#set label, days, tmax, tmin in life event dataframe
lifeevent_data =[('Born',
        day_of_year('19750122') - day_of_year('20160101') + \
        day_of_year('20161231') - day_of_year('20160921') + 2,
        born_set.loc[born_set['DATE'] == 19750122]['TMAX'].values,
        born_set.loc[born_set['DATE'] == 19750122]['TMIN'].values),
                 ('Little league all-star team wins regional championship',
        day_of_year('19880714')-day_of_year('20160101') + \
        day_of_year('20161231')-day_of_year('20160921') + 1,
        little_league_win_set.loc[little_league_win_set['DATE'] == 19880714]['TMAX'].values,
        little_league_win_set.loc[little_league_win_set['DATE'] == 19880714]['TMIN'].values)]

for label, days, tmax, tmin in lifeevent_data:
    plt.scatter(days, tmax, s= 40 , marker="o", color='yellow', edgecolor='k', hatch='|')
    plt.scatter(days, tmin, s= 40 , marker="o", color='yellow', edgecolor='k', hatch='|')
    # annotation for maximum
    ax1.annotate(label, fontsize = 10, fontweight = 'heavy', color = 'k', xy = (days, tmax),
                 xytext = (days, tmax + 20), arrowprops=dict(facecolor='yellow'),
                 horizontalalignment = 'right',verticalalignment = 'top')
    # labels for minimum 
    #ax1.annotate(label, xy=(days, tmin),
    #             xytext=(days, tmin+20),
    #             arrowprops=dict(facecolor='yellow'),
    #             horizontalalignment = 'right',verticalalignment='top')

# Plotting functions                   
minorLocator = MultipleLocator(1)
ax1.xaxis.set_minor_locator(minorLocator) 
plt.title("Temperatures of Ricardo's Lifetime" , fontsize=15)
plt.xlabel(r'Date / Seasons')
plt.ylabel(r'Temperatures(F)')
xvals = [30,120,210,300] # divided roughly into four seasons in 366days
xlabs = ['Fall','Winter','Spring','Summer']
ax1.set_xticks(xvals)
ax1.set_xticklabels(xlabs)
plt.ylim([-50,120])
plt.xlim([0,365])
output_path = os.path.join(output_dir, 'Fig_1')
plt.savefig(output_path)
#plt.show()
plt.close()


'''
------------------------------------------------------------------------
Problem 2. 3D histogram
------------------------------------------------------------------------
'''
# set the dataframe
lipids = pd.read_csv('lipids.csv',skiprows = 3).query('diseased != 0')

# set the number of bins and weights
num_bins = 25
weights = (1 / lipids['chol'].shape[0]) * np.ones_like(lipids['chol'])

# plot 2D histogram
n, bin_cuts, patches = plt.hist(lipids['chol'].values, num_bins, weights=weights)
plt.title('Histogram of Cholesterol ', fontsize=17)
plt.xlabel(r'Level')
plt.ylabel(r'Percent of observations in bin')
output_path = os.path.join(output_dir, 'Fig_2a')
plt.savefig(output_path)
#plt.show()
plt.close()

# find midpoint 
highest_bin = np.argmax(n)
midpoint = (bin_cuts[highest_bin] + bin_cuts[highest_bin + 1]) / 2
print("Problem 2.")
print("2-a. The midpoint of 2d histogram of Cholesterol is", midpoint)
print()

# Problem 2 - b : Plot the 3D histogram
# set the dataframe
lipids = pd.read_csv('lipids.csv', skiprows = 3, index_col = 'diseased')\
.query('diseased !=0')
lipids_chol= lipids['chol']
lipids_trig = lipids['trig']

# Plot 3d histogram using below codes
from mpl_toolkits.mplot3d import Axes3D
'''
--------------------------------------------------------------------
bin_num  = integer > 2, number of bins along each axis
hist     = (bin_num, bin_num) matrix, bin percentages
xedges   = (bin_num+1,) vector, bin edge values in x-dimension
yedges   = (bin_num+1,) vector, bin edge values in y-dimension
x_midp   = (bin_num,) vector, midpoints of bins in x-dimension
y_midp   = (bin_num,) vector, midpoints of bins in y-dimension
elements = integer, total number of 3D histogram bins
xpos     = (bin_num * bin_num) vector, x-coordinates of each bin
ypos     = (bin_num * bin_num) vector, y-coordinates of each bin
zpos     = (bin_num * bin_num) vector, zeros or z-coordinates of
            origin of each bin
dx       = (bin_num,) vector, x-width of each bin
dy       = (bin_num,) vector, y-width of each bin
dz       = (bin_num * bin_num) vector, height of each bin
--------------------------------------------------------------------
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
bin_num = int(25) # 25 equally spaced bins 
hist, xedges, yedges = np.histogram2d(lipids_chol, lipids_trig, bins=bin_num) 
hist = hist / hist.sum()
x_midp = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
y_midp = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])
elements = (len(xedges) - 1) * (len(yedges) - 1)
ypos, xpos = np.meshgrid(y_midp, x_midp)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = (xedges[1] - xedges[0]) * np.ones_like(bin_num)
dy = (yedges[1] - yedges[0]) * np.ones_like(bin_num)
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal', zsort='average')
ax.set_xlabel('Cholesterol Level')
ax.set_ylabel('Trigliceride Level')
ax.set_zlabel('Percent of observations')
plt.title("3D Histogram of 320 individuals' \n Cholesterol and Trigliceride level")
output_path = os.path.join(output_dir, 'Fig_2b')
plt.savefig(output_path)
#plt.show()
plt.close()

print("2-b. The key new characteristics that emerges from the data is Trigliceride. \
Trigliceride has different highest bin compared to Cholesterol and it has lower level \
than Cholesterol. Combined with two data set, many individuals who have evidence of\
heart disease seem to have relatively low Trigliceride level under 200 and\
Cholesterol level from 150 to 250 level.")
print()
print("2-c. As seen in the Fig_2b, the groups might have the highest risk of heart disease\
in the graph have relatively low level of Trigliceride like under 200 level and relatively\
low to mid level of Cholesterol like approximately from 150 to 250 level.")
print()
print()

'''
------------------------------------------------------------------------
Problem 3. Comparing segments of time series
------------------------------------------------------------------------
'''
# set the figure size
fig, ax = plt.subplots(figsize = (12, 6))

# set the dataframe
whole_recession = pd.read_csv('payems.csv',skiprows = 5)

# get the data, year, month data and add it to the dataframe
whole_recession['date'] = pd.to_datetime(whole_recession['date'])
whole_recession['year'] = whole_recession['date'].map(lambda x : x.year)
whole_recession['month'] = whole_recession['date'].map(lambda y: y.month)

# make extraordinary year data to normal year data (like 2029 -> 1929)
whole_recession.ix[whole_recession.year > int(2016), 'year'] \
= whole_recession.ix[whole_recession.year > int(2016), 'year']-100

# to make yearly data to monthly data, generate another 12 repeated data frame
# and set all the values as Nan 
repeat_df = whole_recession[:10]
new_data = repeat_df.loc[np.repeat(repeat_df.index.values, 12)].reset_index()
new_data.payems = np.NaN

# to find the actual job level and set back into original data it is all July of certain year
for i in range(10):
  is_this_year = whole_recession['year'] == new_data.iloc[6 + 12 * i].year
  this_year_payems = whole_recession[is_this_year].payems.values.tolist()
  new_data.set_value(6 + 12 * i,'payems',this_year_payems[0])
monthly_data = whole_recession[10:] # data after 1939
# combine with repeated data and the monthly data 
new_whole_recession = pd.concat([new_data, monthly_data], ignore_index = True)
new_whole_recession = new_whole_recession.drop('index',1)

def normal_series(peakdate):
    '''
    --------------------------------------------------------------------
    Take peakdate information and generate normalized series of certain
    recession periods.
    --------------------------------------------------------------------
    INPUTS:
    string of specific peak month and year

    RETURNS: 
    series that contains normailzed job level of 9 years 
                                     (-1years yo +7years)
    --------------------------------------------------------------------
    '''
    normal_series_lst = []
    date = datetime.strptime(peakdate, '%m/%Y')
    # extract year and month
    first_peak_year, first_peak_month = (date.year,date.month)
    # extract the data that has the same year and month with peak date
    is_recession_year = new_whole_recession['year'] == first_peak_year
    is_recession_month = new_whole_recession['month'] == first_peak_month
    # drop all the rows with None
    not_none = new_whole_recession.payems.notnull()
    peak_index = new_whole_recession[is_recession_year & is_recession_month & not_none].index.tolist()
    peak_value = new_whole_recession[is_recession_year & is_recession_month & not_none].payems.values
    
    # make the x-axis list that contains the data from one year 
    # before peak date to 7 years after peak date (it contains 97 elements ( = 1 + 12 * 8)) 
    for i in range(peak_index[0]-12,peak_index[0] + 7 * 12 + 1):
        if new_whole_recession.iloc[i].payems == None :
            normal_series_lst.append(None)
        else:
            # normalized
            normal_series_lst.append(new_whole_recession.iloc[i].payems / peak_value)
    # for 1929 data, make another one former year list with None and combine with actual data list
    if peakdate == '7/1929':
        normal_series_lst = [None if i in range(peak_index[0] + 6) else actual
        for i, actual in enumerate(normal_series_lst)]
    return np.array(normal_series_lst)


# Plot the whole 14 recession normalized job level graph

# plot 1,2 use the smoothing methods using mask
#1
data_1929 = normal_series('7/1929').astype(np.double)
xaxis_1929=np.array([x for x in range(len(data_1929))]).reshape(len(data_1929),1)
mask1 = np.isfinite(data_1929)
plt.plot(xaxis_1929[mask1], data_1929[mask1], linewidth='4', color ='k',label='07/1929')
#2
data_1937 = np.array(normal_series('7/1937')).astype(np.double)
xaxis_1937=np.array([x for x in range(len(data_1937))]).reshape(len(data_1937),1)
mask2 = np.isfinite(data_1937)
plt.plot(xaxis_1937[mask2], data_1937[mask2], '--', linewidth='1', color ='g',label='07/1937')

#3
data_1945 = normal_series('2/1945')
xaxis_1945=[x for x in range(data_1945.size)]
plt.plot(xaxis_1945, data_1945,'--',linewidth='1', color ='g', label='02/1945')
#4
data_1948 = normal_series('11/1948')
xaxis_1948=[x for x in range(data_1948.size)]
plt.plot(xaxis_1948, data_1948,'.:', linewidth='1.5', color ='brown', label='11/1948')
#5
data_1953 = normal_series('7/1953')
xaxis_1953=[x for x in range(data_1953.size)]
plt.plot(xaxis_1953, data_1953,'--.', linewidth='1', color ='m', label='07/1953')
#6
data_1957 = normal_series('8/1957')
xaxis_1957=[x for x in range(data_1957.size)]
plt.plot(xaxis_1957, data_1957, '-.', linewidth='1', color ='y', label='08/1957')
#7
data_1960 = normal_series('4/1960')
xaxis_1960=[x for x in range(data_1960.size)]
plt.plot(xaxis_1960, data_1960,'.:', linewidth='1', color ='b', label='04/1960')
#8
data_1969 = normal_series('12/1969')
xaxis_1969=[x for x in range(data_1969.size)]
plt.plot(xaxis_1969, data_1969,'--', linewidth='1', color ='teal',label='12/1969')
#9
data_1973 = normal_series('11/1973')
xaxis_1973=[x for x in range(data_1973.size)]
plt.plot(xaxis_1973, data_1973, '-',  linewidth='1', color ='orange',label='11/1973')
#10
data_1980 = normal_series('1/1980')
xaxis_1980=[x for x in range(data_1980.size)]
plt.plot(xaxis_1980, data_1980,'--', linewidth='1', color ='brown',label='01/1980')
#11
data_1981 = normal_series('7/1981')
xaxis_1981=[x for x in range(data_1981.size)]
plt.plot(xaxis_1981, data_1981,'--', linewidth='1', color ='b',label='07/1981')
#12
data_1990 = normal_series('7/1990')
xaxis_1990=[x for x in range(data_1990.size)]
plt.plot(xaxis_1990, data_1990,':', linewidth='1', color ='purple',label='07/1990')
#13
data_2001 = normal_series('3/2001')
xaxis_2001 = [x for x in range(data_2001.size)]
plt.plot(xaxis_2001, data_2001,'-.', linewidth='1', color ='maroon',label='03/2001')
#14
data_2007 = normal_series('12/2007')
xaxis_2007 = [x for x in range(data_2007.size)]
plt.plot(xaxis_2007, data_2007, linewidth='4', color ='red',label='12/2007')

# plot functions 
plt.title('Normalized percentage chage from the peak job levels \n during 14 recessions', fontsize=15)
months_in_8years = 12 * 8
x_vals = np.linspace(0,months_in_8years,9) # evenlty spaced with 9 years 
xticklabs = ['-1yr', 'peak', '+1yr', '+2yr', '+3yr', '+4yr', '+5yr', '+6yr','+7yr']
plt.xticks(x_vals, xticklabs)
plt.xlabel(r'Time from peak')
plt.ylabel(r'Jobs/ peak')
plt.axvline(x=12, color = 'grey', ls = 'dashed', linewidth='1.5')
plt.axhline(y=1, color = 'grey', ls = 'dashed', linewidth='1.5')
plt.legend(bbox_to_anchor=(1, 1), loc = 2, borderaxespad = 1, ncol = 1, fontsize = 9)
output_path = os.path.join(output_dir, 'Fig_3')
plt.savefig(output_path)
#plt.show()
plt.close()

plt.close('all')

print("Problem 3.")
print("3-j. In the short term like the period from peak year to 1 year later,\
there are several U.S recessions worse than the Great Recession in terms of jobs\
beside the Great Depression, but in the long term through 9 years after, other recessions\
beside the Great Depression recovered above the peak year's job level faster than \
the Great Recession. So we can say there are any U.S recessions worse than \
the Great Recession in terms of jobs beside the Great Depression.")
print()
print("3-k. As you seen in the Fig_3 graph, there are any ways in which the Great Recession\
has been worse than the Great Depression in the U.S. The Great Depression job level kept\
fall down for almost 3 years and started to recover slowly and even 9 years after\
the job level could not recover to the peak year job level.")
