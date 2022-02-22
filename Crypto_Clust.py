#j Import Dependencies 
import requests 
import pandas as pd
import matplot.lib.pyplot as plt 
import hvplot.panda
import plotly.express as px 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 

## Create a dataframe 
### Import csv data and use Pandas to create a dataframe that consists ONLY of crypto being traded

data_path = open(".Resources/crypto_data.csv") 
df = pd.read_csv(data_path, index_col=0)
active_crypto_df = df[df.IsTrading.eq(True)]
df_crypto.shape 

### Alter dataframe to remove cryptos with working algo; remove is trading column; Remove rows with Null value; Remove cryptos that have no mining; Remove N/As; 

pd.isna(active_crypto_df['Algorithm']) # Remove anything that doesn't have algos
active_crypto_df = active_crypto_df(['IsTrading'], axis = 1) # Removing IsTrading column 
active_crypto_df = active_crypto_df.dropna(how='any', axis=0) # Dropping rows with null values 
active_crypto_df = active_crypto_df[active_crypto_df.TotalCoinsMined > 0] # Removing rows of crypto that have less than 0 mined coins
active_crypto_df = active_crypto_df[active_crypto_df!='N/A']

### Move CoinName column into its own DF and then drop the CoinName column from the Primary DF
Cryptocurrency_DF = active_crypto_df.filter(['CoinName'], axis=1)
active_crypto_df = active_crypto_df.drop(['CoinName'], axis=1) 

### Create dummy values for text features, drop Algo and ProofType Column, and Standardize the data  
foo_crypto = pd.get_dummies(active_crypto_df['Algorithm'])
foo_dummy = pd.get_dummies(active_crypto_df['ProofType']) 
foo_combined = pd.concat([crypto, dummy], axis=1)
df = active_crypto_df.merge(combined, left_index=True, right_index=True) 
df = df.drop(['Algorithm', 'ProofType'], axis=1)
scale_df = StandardScaler().fit_transform(df) 

## Clean data for PCA 
### Reduce Dimensions for better Fitted Data 

pca = PCA(n_components=3) 
PCA_data = pca.fit_transform(df_scaled) 
PCA_df = pd.DataFrame(data=PCA_data, columns=['PC1', 'PC2', 'PC3'], index=Cryptocurrency_DF.index) 

## Cluster Crypto w/ Kmeans
### Find K with the Elbow Curve 

constant = []
k = list(range(1,11)) 
for i in k: 
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(PCA_data) 
    constant.append(km.inertia) 
Elbow_Curve_Data = {"k" :k,"Inertia":constant}
Elbow = pd.DataFrame(elbow_data) 
Elbow.hvplot(x="k", y="inertia", xticks=k, title="Cryptocurrency Elbow Curve") 

### Fit the model 

kmeans = Kmeans(n_clusters=4, random_state=0) # Initalize K means model
kmeans.fit(PCA_df) 
