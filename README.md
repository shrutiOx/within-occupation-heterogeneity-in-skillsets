Understanding within-occupation heterogeneity in skillsets using large online job vacancy data

Data and method
 • 20 million observations for one year, 2019 (we use 300k for tests)
 
 • From Lightcast, a private provider
 
 • Occupation, metropolitan area, industry, and 25 skills described for each 
observation, including 8 digital skills observation

 • Try on a small sample:
 
   • Keep only popular occupations (n > 500) → 250k observations,
   
   • Formed skill clusters with different methods and parameters (k-means with different 
number of clusters, DBSCANS, hierarchical clustering)

   • Form skill clusters clusters for technical skills,
   
   • Choose the optimal clustering method (elbow method, Davies-Bouldin, Calinski
Harabasz),

   • Each cluster gives us a unique combination of skills,
   
   • Each occupation contains different skillsets.
