# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:03:33 2024

@author: ADMIN
"""

'''Experiment with 300K 2019'''

'''
    1.Process 300K data like 100K
    2.Get clusters with all 25 skills using k-means.
    3.Use 2 kind of algos with K-means and check how they result.
    4.Check if propensity calculation is correct.
    5.Get cluster vs skill sets in descending order of their propensities. Store this.
    6.Get cluster number vs soc6.
    7.Now get a pandas df (later save to excel) where you have soc6 vs cluster (one that is in max amount).
    8.Now get correlation heatmap and clustermaps.
    9.Now do evaluation between 2 kind of algos used in k-means. Make sure u definitively understand the difference between them.
'''






'''importing libraries'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

import pickle 

'''take 2019 data'''


path = "jobskills_clean_2019-002.dta"
i=0

with pd.read_stata(path, chunksize=100000) as itr:
    for chunk in itr:
        if i  == 0:
            df_2019 = chunk
        else:
            df_2019 = pd.concat([df_2019, chunk])
        i += 1


descr = df_2019.describe()

print(descr )

 
df_2019.head()

'''List all the coloumns'''

listi = list(df_2019.columns)

'''There are 25 skill categories above. Now we will drop all other coloumns and keep the skill categories (25) which will be used to do unsupervised learning (cluster formation) and we will keep soc6,  edu, msa'''


# Select the coloumns needed
df2 = df_2019[['soc6','msa','edu',
 'social',
 'prob_solv',
 'decision',
 'crit_think',
 'think',
 'analytic',
 'business_analysis',
 'noncog',
 'mgmt',
 'finance',
 'service',
 'computer',
 'admin_support',
 'tech_support',
 'gen_soft',
 'engineer',
 'products',
 'project_mgmt',
 'writing',
 'creativ',
 'bus_sys',
 'database',
 'data_analysis',
 'ML_AI',
 'special_soft',
 'year']]



'''We drop all rows that has null entry. '''

df2=df2.dropna() 

'''Making sure all are 2019 data only'''

df2 =  df2[ df2['year'] == 2019]


'''Now at first we use  groupby to check how many rows (entries) are there per soc6 category'''

grp=df2.groupby('soc6').count()

'''We want to keep only those soc6 which has more than 500 entries. It will be difficult to form clusters otherwise at a later stage. So we are keeping only those occupations i,e, soc6 which are really relevant i.e. has many entries in the dataset'''


grp2 =  grp[grp['social'] > 500]

grp2 = grp2.reset_index()
list(grp2.columns)

soc_list = grp2['soc6'].tolist()

'''df3 is the cleaned data on which we will work. It has all the required coloumns only  and only those soc6 whose number of entries  is greater than 500  and year = 2019.'''


df3 = df2[df2['soc6'].isin(soc_list)]

'''save this data for later purpose'''

df3.to_stata("Only2019cleaned.dta")



'''Let's suppose we form a wholesome cluster from all the 25 coloumns (skills) of all the rows  of df3. Thus each can say that look  all soc6 categories (from soc_list) together can be grouped into these clusters .'''



'''K-Means clustering, elkan method'''
'''Elkan's algorithm is a variation of Lloyd alternate optimization algorithm (Lloyd's algorithm) that uses the triangular inequality to avoid many distance calculations when assigning points to clusters.'''
'''
ELKAN Method (Acceleration for K-means)

Purpose:  The ELKAN method is an optimization technique that significantly speeds up the K-means algorithm by reducing redundant distance calculations.

Key Idea: It uses the triangle inequality to establish lower and upper bounds on the distances between data points and cluster centers.

How it Works:

Maintain Distance Bounds:  For each cluster, keep track of the minimum and maximum distance between the cluster centroid and any data point previously assigned to that cluster.

Utilizing the Triangle Inequality:

Lower Bound: If the distance between a data point and another cluster's centroid is greater than the sum of that centroid's maximum distance (from its cluster members) and the distance between the two centroids, we can skip calculating the exact distance to that centroid. The point can't be closer to that other centroid.
Upper Bound: Track a running upper bound on the minimum distance between a data point and any centroid. If the distance between a data point and a centroid is less than this upper bound, calculate the exact distance and potentially reassign the point to that cluster.
'''                                                                                  



'''Now we create another dataframe to hold only the 25 skill categories which nees clustering'''

df4 = df3[['social',
 'prob_solv',
 'decision',
 'crit_think',
 'think',
 'analytic',
 'business_analysis',
 'noncog',
 'mgmt',
 'finance',
 'service',
 'computer',
 'admin_support',
 'tech_support',
 'gen_soft',
 'engineer',
 'products',
 'project_mgmt',
 'writing',
 'creativ',
 'bus_sys',
 'database',
 'data_analysis',
 'ML_AI',
 'special_soft']]

'''Our features under consideration'''

features = df4.columns




'''Let us try the elbow method to find optimal clusters.For each k, calculate the Within-Cluster Sum of Squares (WCSS). This measures how spread out the points are within each cluster. Lower WCSS indicates tighter clusters.As we increase k, the WCSS typically keeps decreasing. We intend to look for the point where the decrease in WCSS starts to slow down significantly. This is the "elbow" of the curve. '''

wcss = []
for i in range(1, 26):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df4[features])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 26), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('Elbowcluster1')



'''Based on elbow method observation, it seems 6 should be an appropiate number of clusters'''

'''WE DO K-MEANS CLUSTERING NOW'''

n_clusters = 6

# Create a KMeans model
kmeans = KMeans(n_clusters=n_clusters,algorithm='elkan')

# Fit the model to the  data
#kmeans.fit(scaled_data)
kmeans.fit(df4[features])

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Add cluster labels to the original data
df4['cluster'] = cluster_labels

# Print cluster information (optional)
print(df4['cluster'].value_counts()) #this indicates how many rows corresponds to each cluster

'''Analysis of K-Means results and visualization'''

#df4.to_excel('clusterraw.xlsx')  # this excel gives you information of how each row corresponds to each clusters 

'''Mapping soc6 to this now'''

df5 = df4
df5['soc6'] = df3['soc6']

#df5.to_excel('clusterpersoc.xlsx')  # this excel gives you information of how each row corresponds to each clusters and each soc5 corresponding with it

'''CRITICAL EXPLAINATION - Right now, we have formed clusters per row of the matrix i.e. each row which corresponds to each soc6 has a defined cluster now and this is formed from the 25 skills only. However 25 skills were coloumns and the cluster information is assigned against each row of the data, as per standard procedure. So each cluster indicates some RELATION to skill categories (25 ) now. But we need to do further analysis to exactly tell, WHICH SKILLS FALLS UNDER WHICH CLUSTERS. Let's do that now.'''


grpcluster = df5.groupby('cluster', axis=0)#we are grouping clusters. So number of groups = 6 = num of clusters  This group is not done yet.

'''**Now we simply use mean as aggregate to complete  this grouping. This will give us number of SKILLS divided by the cluster size in each cluster group (mean). Next you normalize the skills (sum all the skills along each row by which you divide each  skills). So you will see the propensity of each of the 25 skills  in each clusters. This will help us to inform our readers/policy makers later when we have the mapping between soc6 vs clustertype 1, that by what percentage do that occupation type need so and so skills. There are overlapping and this is normal, so giving propensity might be the best policy to avoid any internal biaseness.'''


grpnew1 = grpcluster.mean() #

grpnew1 = grpnew1.drop(['soc6'],axis=1)# this makes analysis confusing

grpnew = grpnew1.div(grpnew1.sum(axis=1), axis=0)



grpnew.to_excel('clustertoskillall.xlsx')

'''**Which skill sets map to each clusters ?. Below code sorts each rows and then forms a dictionary which gives CLUSTER vs SKILLs in decending order of their propensities. Instead of taking juts  10 top skills per cluster, we can inform about the propensity to make a safe better informative AI model.'''



listi=[]

dicti = {}

for j in range(0,6):
    listi=grpnew.iloc[j] 
    listi=listi.sort_values(ascending=False)
    dicti.update({j:listi})
    #print(listi)

'''Saving this dictionary and re-loading to check'''



with open('clusterTOskillmapsorted.pkl', 'wb') as f:
    pickle.dump(dicti, f)
        
with open('clusterTOskillmapsorted.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
 
print('Association of all skills')
print(loaded_dict)


'''Now we can add the soc6 info to this again to make it more interesting. Just group by soc6 and cluster'''

grpcluster2 = df5.groupby(['soc6','cluster'], axis=0)

'''Now you get the count of each clusters per soc5. For each soc6, we can consider the cluster that has highest counts, which indicates its occurance in that soc5 group.'''

grpnew2 = grpcluster2['cluster'].count()



grpnewPD=grpnew2.to_frame()

grpnewPD=grpnewPD.rename(columns={"cluster": "clusternum"})
grpnewPD.to_excel('clustertosoccount.xlsx')  

#grpnew21 = grpcluster2['cluster'].count().sort_values(ascending=False).drop_duplicates()

grpnew21 = grpnewPD.groupby(['soc6'], axis=0)['clusternum'].max() #getting the maximum number of cluster per soc6 level 
grpnew21=grpnew21.to_frame()

grpnew211 = grpnewPD.groupby(['soc6'], axis=0)['clusternum'].idxmax()#identifying soc6 vs the cluster which is in maximum amount for that sco6

grpnew211=grpnew211.to_frame()
grpnew211['max values'] = grpnew21['clusternum']

grpnew211.to_excel('socvsmaxcluster.xlsx')  

#grpnew211_2019=pd.read_excel('grpclustersocnew.xlsx')  



'''AT THIS POINT YOU HAVE CLUSTER TO SKILL MAPPING AND SOC5 TO CLUSTER MAPPING AS WELL'''


#sns.heatmap(grpnew,cmap='magma',linecolor='white',linewidths=2,annot=True)

'''Let us plot the cluster to skill mapping with clustermap(heirarchical agglomerative clustering) to see how this looks'''
cluster_plot=sns.clustermap(grpnew,cmap='coolwarm',standard_scale=1)


cluster_plot.savefig("outclustertype1.png") 



'''create a heatmap'''

# Create correlation matrix
corr_matrix = grpnew.corr()

# Set size of visualization
plt.figure(figsize=(20, 20))

corrplot=sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=3)

plt.title('Correlation Matrix Heat Map')
plt.show()
plt.savefig('corrclustertype1.png', dpi=300)



'''calculating the efficiency for algo-elkan and cluster number-6'''

'''
Davies-Bouldin Index (DBI)

Focus: Measures the similarity between clusters themselves. This focuses on both compactness within a cluster and good separation between clusters.
Calculation:
For each cluster, calculate the average distance of all points within that cluster to its centroid. This is a measure of intra-cluster dispersion (compactness).
For each pair of clusters, calculate a similarity measure based on the sum of their intra-cluster dispersions divided by the distance between their centroids.
Find the maximum similarity measure for each cluster and average those maximum values over all clusters. This yields the DBI.


Interpretation:

Lower DBI is better. A smaller DBI score indicates compact clusters that are well-separated from each other.



Calinski-Harabasz Index (CHI)

Focus: Measures the ratio of between-cluster dispersion to within-cluster dispersion.
Calculation:
Between-cluster dispersion: Measure of how scattered the cluster centroids are in relation to the overall dataset centroid.
Within-cluster dispersion: Average of how tightly packed each cluster is around its own center.
CHI: The ratio of between-cluster dispersion to within-cluster dispersion.

Interpretation:
    
Higher CHI is better. A larger CHI score suggests well-separated, dense clusters.
'''


# Calculate metrics
davies_boulding = davies_bouldin_score(df4[features], cluster_labels)
calinski_harabasz = calinski_harabasz_score(df4[features], cluster_labels)


print(f'Davies-Bouldin Index cluster = 6,type=1,algo-elkan: {davies_boulding}')
print(f'Calinski-Harabasz Index cluster = 6,type=1,algo-elkan: {calinski_harabasz}')


'''
# **Grouping and Aggregation - propensity calculation**
'''

def calculate_proportions(df):
    grouped = df.groupby(['soc6', 'cluster'],axis=0)  # Group by last two columns
    counts = grouped['cluster'].count()  # Count occurrences in each group
    #print(counts)
    total_counts = counts.groupby(['soc6']).sum()  # Sum counts within each 'soc6' group
    #print(total_counts)
    return counts / total_counts

result = calculate_proportions(df5.copy())  
print(result)

result.to_excel('Propensitysocvscluster.xlsx')