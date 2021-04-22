
from datetime import date
import pandas as pd


#visualizations
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import seaborn as sns
#%matplotlib inline
def save():
    with open('Mall_Customers.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()


import sklearn.cluster as cluster

sns.set_style()
plt.style.use('fivethirtyeight')


import streamlit as st

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from csv import writer



st.set_option('deprecation.showPyplotGlobalUse', False)

#@st.cache
#URL = "A:\\Fde_final_app\\Mall_Customers.csv"
customer = pd.read_csv('Sample_Customers.csv')
#st.write(customer.head())
#customerdata = customer[['RFMScore']].astype(int)

customer.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)

st.title("Customer segmentation")


st.sidebar.title("Our Model")

st.sidebar.info("""
	This is a machine learning project, 
	Where we analyze and cluster a customers data
	based on their spending habits and their income.
	""")

st.markdown("""
	## About Data
	This data contains the basic information (ID, age, gender, income, spending score) about the customers
	""")


st.cache(persist = True)
st.sidebar.markdown("""
	## Customer Data
	""")
if st.sidebar.checkbox("RFM ANALYSIS"):
    rfm1 = pd.read_csv('Mall_Customers2.csv')
    st.write(rfm1.head())
    recency = rfm1.groupby(by='CustomerID', as_index=False)['Date'].max()
    recency.columns = ['CustomerID','LastPurshaceDate']
    st.write(recency)
    recency['Date'] = pd.DatetimeIndex(recency['LastPurshaceDate']).date
    today = date.today()
    st.write(today)
    recency['now'] = today
    st.write(recency)
    recency['recency'] = (recency['now'] - recency['Date']).dt.days
    st.write(recency)
    monetary = rfm1.groupby(by='CustomerID',as_index=False).agg({'AmountSpent': 'sum'})
    monetary.columns = ['CustomerID','Monetary']
    st.write(monetary)
    rfm2 = rfm1
    frequency = rfm2.groupby(by=['CustomerID'], as_index=False)['Annual Income (k$)'].count()
    frequency.columns = ['CustomerID','Frequency']
    st.write(frequency)
    recency.drop('LastPurshaceDate',axis=1,inplace=True)
    recency.drop('Date',axis=1,inplace=True)
    recency.drop('now',axis=1,inplace=True)
    st.write(recency)

    rfm = recency.merge(frequency,on='CustomerID')
    rfm = rfm.merge(monetary,on='CustomerID')
    rfm.set_index('CustomerID',inplace=True)
    st.write(rfm)

    quantiles = rfm.quantile(q=[0.25,0.5,0.75])
    st.write(quantiles.head())

    quantiles.to_dict()

    def RScore(x,p,d):
        if x <= d[p][0.25]:
         return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]:
            return 2
        else:
            return 1
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
    def FMScore(x,p,d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]:
            return 3
        else:
            return 4

    rfm_segmentation = rfm
    rfm_segmentation['R_Quartile'] = rfm_segmentation['recency'].apply(RScore, args=('recency',quantiles,))
    rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
    rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))
    st.write(rfm_segmentation)

    rfm_segmentation['RFMScore'] = (((rfm_segmentation.R_Quartile + rfm_segmentation.F_Quartile + rfm_segmentation.M_Quartile)/3)*25).astype(int)
    st.write(rfm_segmentation)

    df = pd.read_csv("Mall_Customers.csv")
    df["RFMScore"] = rfm_segmentation['RFMScore']
    df.to_csv("Mall_Customers.csv", index = False)





if st.sidebar.checkbox("RAW DATA"):
    st.subheader("Raw Data")
    st.write(customer)
    if st.button("ADD CUSTOMER"):
        User_customer = st.text_input("Customer_ID")
        user_age = st.text_input("Age")
        user_gender = st.text_input("Gender")
        user_Income = st.text_input("Annnual_Income")
        user_Score = st.text_input("Spending_Score")


        List=[User_customer,user_gender,user_age,user_Income,user_Score]

        if st.button("Submit"):
            save()


# This shows male and female ratio


st.sidebar.markdown("""
	## Data Analysis
	""")

st.markdown("""
	## Data Analysis
	 Data is one of the important features of every organization because it helps business leaders to make decisions based on facts, statistical numbers and trends. Due to this growing scope of data, data science came into picture which is a multidisciplinary field. It uses scientific approaches, procedure, algorithms, and framework to extract the knowledge and insight from a huge amount of data.
	""")

if st.sidebar.checkbox("Data Analysis"):
    st.subheader("Gender Ratio")
    labels = ['Female', 'Male']
    size = customer['Gender'].value_counts()
    colors = ['lightgreen', 'lightblue']
    explode = [0, 0.1]

    plt.rcParams['figure.figsize'] = (4, 4)
    plt.title('Gender', fontsize = 10)
    plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
    plt.axis('off')
    plt.legend()
    st.pyplot()

    st.subheader("Age countplot")
    plt.figure(figsize = (22,14))
    ax = sns.countplot(x="Age", hue="Gender", data=customer)
    st.pyplot()

    st.subheader("Income Countplot")
    plt.figure(figsize = (22,14))
    ax = sns.countplot(x="Income", hue="Gender", data=customer)
    st.pyplot()

    st.subheader("Spending Score Countplot")
    plt.figure(figsize = (22,14))
    ax = sns.countplot(y="Spending_Score", hue="Gender", data=customer)
    st.pyplot()

    st.subheader("Spending_Score vs Income boxplot")
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.boxplot(y=customer["Spending_Score"], color="lightblue")
    plt.subplot(1,2,2)
    sns.boxplot(y=customer["Income"], color="lightgreen")
    st.pyplot()

    st.subheader("Pairplot")
    sns.pairplot(customer, vars= ['Age', 'Income', 'Spending_Score'], hue="Gender")
    fig = plt.gcf()
    fig.set_size_inches(22,14)
    st.pyplot()




# Cluster for income V/S Spending score
st.markdown(""" 
	## *K Means Clustering Model*
	K-means clustering algorithm computes the centroids and iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. It is also called flat clustering algorithm. The number of clusters identified from data by algorithm is represented by ‘K’ in K-means.
	""")
st.sidebar.markdown("""
	## Group Clustering by Age & Income
	""")

if st.sidebar.checkbox("Group Clustering"):
    st.subheader("Group Clustering for Age Vs Spending score")
    customer_short = customer[['Spending_Score','Income']].astype(int)
    K=range(1,12)
    wss = []
    for k in K:
        kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
        kmeans=kmeans.fit(customer_short)
        wss_iter = kmeans.inertia_
        wss.append(wss_iter)
    mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})


    sns.lineplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="+")
    plt.title("Spending score x Income")
    st.subheader("Elbow plot for Income V/S Spending Score")
    fig = plt.gcf()
    fig.set_size_inches(22,14)
    st.pyplot()
    # We get 5 Clusters

    # Model for spending score V/S Income
    kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")
    kmeans = kmeans.fit(customer[['Spending_Score','Income']])

    customer['Clusters'] = kmeans.labels_

    customer.to_csv('mallClusters.csv', index = False)
    st.subheader("Clusters Pairplot for Income V/S Spending Score")
    sns.pairplot(customer, vars= ['Age', 'Income', 'Spending_Score'], hue="Clusters", height=4)
    st.pyplot()

    y = customer.iloc[:, [2, 4]].values

    from sklearn.cluster import KMeans

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(y)
        wcss.append(kmeans.inertia_)
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    ymeans = kmeans.fit_predict(y)

    plt.rcParams['figure.figsize'] = (22,14)
    plt.title('Cluster of Annual Income', fontsize = 30)

    plt.scatter(y[ymeans == 0, 0], y[ymeans == 0, 1], s = 100, c = 'pink', label = 'Priority Customers' )
    plt.scatter(y[ymeans == 1, 0], y[ymeans == 1, 1], s = 100, c = 'orange', label = 'Target Customers(High income)')
    plt.scatter(y[ymeans == 2, 0], y[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Usual Customers')
    plt.scatter(y[ymeans == 3, 0], y[ymeans == 3, 1], s = 100, c = 'yellow', label = 'Target Customers(Less Income)')
    plt.scatter(y[ymeans == 4, 0], y[ymeans == 4, 1], s = 100, c = 'blue', label = 'Moderately spending customers')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'Black', marker ='*')
    st.subheader("Cluster plot for Income V/S Spending Score")
    plt.style.use('fivethirtyeight')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid()
    st.pyplot()

    # Cluster for Age V/S Spending score

    st.subheader("Group Clustering for Annual Income Vs Spending score")
    customer_range = customer[['Spending_Score','Age']]

    K=range(1,12)
    wss = []
    for k in K:
        kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
        kmeans=kmeans.fit(customer_range)
        wss_iter = kmeans.inertia_
        wss.append(wss_iter)

    mycenters1 = pd.DataFrame({'Clusters1' : K, 'WSS' : wss})
    st.subheader("Elbow plot for Age V/S Spending Score")
    sns.lineplot(x = 'Clusters1', y = 'WSS', data = mycenters1, marker="+")
    plt.title("Spending score x Age")

    fig = plt.gcf()
    fig.set_size_inches(22,14)
    # We get 4 Clusters
    st.pyplot()

    x = customer.iloc[:, [2, 4]].values
    from sklearn.cluster import KMeans

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    ymeans = kmeans.fit_predict(x)

    plt.rcParams['figure.figsize'] = (22,14)
    plt.title('Cluster of Ages', fontsize = 30)
    st.subheader("Cluster plot for Age V/S Spending Score")
    plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )
    plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')
    plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')
    plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'yellow', label = 'Target Customers(Old)')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')

    plt.style.use('fivethirtyeight')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid()
    st.pyplot()



st.sidebar.markdown("""
	## Two Step Clustering
	""")

if st.sidebar.checkbox("Two Step Clustering Algorithm"):
    st.subheader("Two step clustering by Age")
    st.subheader("Dendrogram to find out clusters")
    X = customer.iloc[:,[2,4]].values
    plt.figure(figsize=(15,6))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.hlines(y=190, xmin=0, xmax=2000, lw=3, linestyles='--')
    plt.text(x=900, y=220, s='Horizontal line crossing 5 vertical lines', fontsize=20)
    #plt.grid(True)
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    st.pyplot()

    X = customer.iloc[:,[2,4]].values
    hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    plt.figure(figsize=(12,7))
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful-Cluster1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard-Cluster2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target group-Cluster3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'orange', label = 'Careless-Cluster4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible-Cluster5')
    plt.title('Clustering of customers', fontsize=20)
    plt.xlabel('Age', fontsize=16)
    plt.ylabel('Spending Score (1-100)', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    st.subheader("Hierarchical Clustering Clusters")
    plt.axhspan(ymin=60, ymax=100, xmin=0.4, xmax=0.96, alpha=0.3, color='yellow')
    st.pyplot()

    st.subheader("Two step clustering by Annual Income")
    st.subheader("Dendrogram to find out clusters")
    X = customer.iloc[:,[3,4]].values
    plt.figure(figsize=(15,6))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.hlines(y=190, xmin=0, xmax=2000, lw=3, linestyles='--')
    plt.text(x=900, y=220, s='Horizontal line crossing 5 vertical lines', fontsize=20)
    #plt.grid(True)
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    st.pyplot()

    X = customer.iloc[:,[3,4]].values
    hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    plt.figure(figsize=(12,7))
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful-Cluster1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard-Cluster2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target group-Cluster3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'orange', label = 'Careless-Cluster4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible-Cluster5')
    plt.title('Clustering of customers', fontsize=20)
    plt.xlabel('Annual Income (k$)', fontsize=16)
    plt.ylabel('Spending Score (1-100)', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    st.subheader("Hierarchical Clustering Clusters")
    plt.axhspan(ymin=60, ymax=100, xmin=0.4, xmax=0.96, alpha=0.3, color='yellow')
    st.pyplot()
