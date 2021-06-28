
# coding: utf-8

# Student name (Student ID): Lai Siu Kwok (12354991), Lo Shi Sam (12339625), Hung Chun Kwong (12358793)

# # Problem Definition

# Heart failure is a serious disease that is difficult to cope with due to its sudden onset and there are many people who die of heart failure every year. In view of this, the aim of this project is to use machine learning techniques such as tests with different algorithms, to find out which algorithms perform well which is the highest accuracy, in this case. Therefore, the best algorithms can be used to predict the survival of patients with heart failure in order to help people who need the medication the most for reducing the number of deaths from heart failure.

# # Introduction

# We will use K-fold cross-validation to compare 4 algorithms, respectively are logistic regression (LR), K Nearest Neighbors (KNN) Naive Bayes (NB), and K-Mean (KM). First, we will divide the dataset into training and testing set as a ratio of 7:3. Second, the normalization techniques will be deployed because of the different types of features. Third, the feature selection will be performed for finding the most significant two features in order to suit the algorithms. Forth, the visualization techniques of the output will give a clearer image for the explanation. Finally, The cross-validation (CV) scores obtained can then be used as criteria for choosing the best parameter of the models, also the model selection for future work in the end.

# # Library

# In[1]:


from pandas import read_csv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# # Dataset import

# In[2]:


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
del df['time'] #The column 'time' is day/time property, not a useful feature
print("Dimention: ", df.shape)


# ## Find if missing values is present

# In[3]:


print(df.isnull().sum())
print("\nNo missing value in dataset")


# # Data Description

# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


sns.countplot(x="DEATH_EVENT", data = df)
pyplot.title("Distribution of DEATH_EVENT")


# In[7]:


pyplot.style.use('ggplot')
pyplot.title("Distribution of age")
pyplot.hist(df['age'], bins=30)
print()


# In[8]:


sns.countplot(x="anaemia", data = df)
pyplot.title("Distribution of anaemia")
print()


# In[9]:


pyplot.style.use('ggplot')
pyplot.hist(df['creatinine_phosphokinase'], bins=20)
pyplot.title("Distribution of creatinine_phosphokinase")
print()


# In[10]:


sns.countplot(x="diabetes", data = df)
pyplot.title("Distribution of diabetes")
print()


# In[11]:


pyplot.style.use('ggplot')
pyplot.hist(df['ejection_fraction'], bins=20)
pyplot.title("Distribution of ejection_fraction")
print()


# In[12]:


sns.countplot(x="high_blood_pressure", data = df)
pyplot.title("Distribution of high_blood_pressure")
print()


# In[13]:


pyplot.style.use('ggplot')
pyplot.hist(df['platelets'], bins=30)
pyplot.title("Distribution of platelets")
print()


# In[14]:


pyplot.style.use('ggplot')
pyplot.hist(df['serum_creatinine'], bins=30)
pyplot.title("Distribution of serum_creatinine")
print()


# In[15]:


pyplot.style.use('ggplot')
pyplot.hist(df['serum_sodium'], bins=30)
pyplot.title("Distribution of serum_sodium")
print()


# In[16]:


sns.countplot(x="sex", data = df)
pyplot.title("Distribution of sex")
print()


# In[17]:


sns.countplot(x="smoking", data = df)
pyplot.title("Distribution of smoking")
print()


# ## Correlation matrix

# In[18]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# # Interpretation

# From the above plots, we can see there are no any abnormal phenomenon and distribution about the features. On the other hand, the correlation plot indicates that most of the features have a correlation relationship to the target. Therefore, we can go for conducting our proposed algorithms.

# # Preprocessing

# ## Normalization

# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df.drop('DEATH_EVENT',axis=1))
scaled_features = scaler.transform(df.drop('DEATH_EVENT',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# ## Split data into training / testing set

# In[20]:


array = df_feat.values
X = array[:,:]
Y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) 

print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)


# ## Feature selection

# In[21]:


from sklearn.feature_selection import chi2, SelectKBest, f_classif
ft = SelectKBest(chi2, k = 2).fit(X_train, y_train)
print('Score: ', ft.scores_)
print(df_feat.columns)


# In[22]:


ft = SelectKBest(f_classif, k= 2).fit(X_train, y_train)
print('Score: ', ft.scores_)
print(df_feat.columns)


# ### Interpretation

# From the result of feature selection, in order to fit the algorithms, we have to get two most important features. Then, we can notice that the features of 'ejection_fraction' and 'serum_creatinine' having the highest mark from the two criteria. Thus, 'ejection_fraction' and 'serum_creatinine' will be selected.

# ## Remap the X set with correlated features

# In[23]:


X_train = X_train[:,(4,7)]
X_test = X_test[:,(4,7)]
print("X_train:\n",X_train.shape)
print("X_test:\n",X_test.shape)

print("y_train:\n",y_train.shape)
print("y_test:\n",y_test.shape)


# # Algorithm implementation

# ## Logistic Regression
# ### Logistic Regression with Cross-Validation

# In[24]:


from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                           cv=10,
                           max_iter=1000,
                           random_state=0,
                           penalty='l2').fit(X_train, y_train)

print("Training Accuracy:", clf.score(X_train, y_train))
print("Coeff:\n", clf.coef_)
#print("The detail of iter across every class:\n", clf.n_iter_)
print("The best C across every class:", clf.C_)


# ### Logistic Regression with best C

# In[25]:


from sklearn import metrics
import seaborn as sn

logreg_fit = LogisticRegression(C=float(clf.C_), max_iter=1000)
logreg_fit = logreg_fit.fit(X_train, y_train)
y_pred = logreg_fit.predict(X_test)


# ### Model Evaluation 

# In[26]:


scores = []
for i in range(1, 1000):
    logreg = LogisticRegression(C=float(clf.C_), max_iter=i)
    logreg_trained = logreg.fit(X_train, y_train)
    scores.append(logreg_trained.score(X_test, y_test))

pyplot.plot(range(1, 50), scores[1:50], marker='o')
pyplot.title("Detail of iter")
pyplot.xlabel('Number iter')
pyplot.ylabel('Accuracy')
pyplot.tight_layout()
pyplot.show()
print("The graph indicates that the # of iter is within the range")


# In[27]:


import matplotlib.pyplot as plt
logreg_best = LogisticRegression(C=float(clf.C_), max_iter=1000)
logreg_fit = logreg_best.fit(X_train, y_train)

h = .02
X_min, X_max = X[:, 4].min() - .05, X[:, 4].max() + .05
y_min, y_max = X[:, 7].min() - .05, X[:, 7].max() + .05
xx, yy = np.meshgrid(np.arange(X_min, X_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6, 4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 4], X[:, 7], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('ejection_fraction')
plt.ylabel('serum_creatinine')
plt.title('Decision boundaries with Logistic Regression')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# In[28]:



print(classification_report(y_test, y_pred))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
pyplot.show()
LR_acc = metrics.accuracy_score(y_test, y_pred)
print('Testing set accuracy: ',LR_acc)


# ## KNN

# ### KNN with Cross-Validation

# In[29]:


error_rate = []
acc = []
scoresCV = []
for i in range(1,40,2): # 1,3,5,7,9 ...... Law of thumb
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
    #cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    scores = metrics.accuracy_score(y_test, pred_i)
    acc.append(scores.mean())
  

pyplot.figure(figsize=(10,6))
pyplot.plot(range(1,40,2),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
pyplot.title('Error Rate vs. K Value')
pyplot.xlabel('K')
pyplot.ylabel('Error Rate')
    
pyplot.figure(figsize=(10,6))
pyplot.plot(range(1,40,2),acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
pyplot.title('Accuracy vs. K Value')
pyplot.xlabel('K')
pyplot.ylabel('Accuracy')

print("Error Rate suggests : 7, 11, 13")
print("Accuracy Rate suggests : 7, 11, 13")


# ### Hyperparameter Tuning

# In[30]:


n_neighbors = [7,11,13]
grid_params = { 'n_neighbors' : n_neighbors,
               'metric' : ['minkowski','euclidean','manhattan']}
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=10, n_jobs = -1)
g_res = gs.fit(X_train, y_train)
print(g_res.best_score_)
print(g_res.best_params_)
#print(g_res.best_index_)
# use the best hyperparameters
knn_best = KNeighborsClassifier(n_neighbors=n_neighbors[int(g_res.best_index_)], metric = 'minkowski')
knn_best.fit(X_train, y_train)

print("The best K is", n_neighbors[int(g_res.best_index_)])


# ### Model Evaluation 
# #### KNN with best K

# In[31]:


y_pred = knn_best.predict(X_test)

print(classification_report(y_test, y_pred))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
pyplot.show()

KNN_acc = metrics.accuracy_score(y_test, y_pred)
print('Testing set accuracy: ', KNN_acc)


# ## Naïve Bayes Classifier

# In[32]:


model = GaussianNB()
trained_model = model.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)


# ### Model Evaluation 

# In[33]:



print(classification_report(y_test, y_pred))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
pyplot.show()

NB_acc = metrics.accuracy_score(y_test, y_pred)
print('Testing set accuracy: ', NB_acc)


# ## K-means Clustering

# In[34]:


colors=np.array(["red", "blue"])
plt.xlabel('ejection_fraction')
plt.ylabel('serum_creatinine')
plt.title('Distrubution')

plt.scatter(X[:, 4], X[:, 7], c=colors[Y], s=50)
for label, c in enumerate(colors):
    plt.scatter([], [], c=c, label=str(label))
plt.legend();


# ### Model fitting

# In[35]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, 
            init='random', 
            n_init=100, 
            max_iter=300,
            tol=1e-04, # stop criteria
            random_state=0)

km_fit = km.fit(X_train,y_train)
km_pred = km_fit.predict(X_test)

#print(help(km_pred))
#print(km_fit.inertia_)
print("Centroid 1:{}\nCentroid 2:{}".format(km_fit.cluster_centers_[0],km_fit.cluster_centers_[1]))
print("# of Iter:",km_fit.n_iter_)


# In[36]:


y_km = km.fit_predict(X_train)

plt.scatter(X_train[y_km == 0, 0],
            X_train[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
plt.scatter(X_train[y_km == 1, 0],
            X_train[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()

plt.show()


# ### Model Evaluation

# In[37]:



print(classification_report(y_test, km_pred))

confusion_matrix = pd.crosstab(y_test, km_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
pyplot.show()

KM_acc = metrics.accuracy_score(y_test, km_pred)
print('Testing set accuracy: ',KM_acc)



# # Algorithm Comparison

# In[38]:


names = ['LR','KNN','NB','KM']

models = []
models.append(('LR', logreg_best))
models.append(('KNN', knn_best))
models.append(('NB', GaussianNB()))
models.append(('KM', km))

results = []
scoring = 'accuracy'

for name, model in models:

    model_fit = model.fit(X_train,y_train)
    cv_results = cross_val_score(model_fit, X_test, y_test, cv=10, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    pt = "%s:\tMean:%f\tStd:%f" % (name, cv_results.mean(), cv_results.std())
    print(pt)

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # Reference

# ### 1. Kevin(2016). A Complete Guide to K-Nearest-Neighbors with Applications in Python and R. Retrieved from https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/ 
# ### 2. Arvai(2020). K-Means Clustering in Python: A Practical Guide. Retrieved from https://realpython.com/k-means-clustering-python/
# ### 3. scikit-learn.org(2020). Confusion matrix. Retrieved from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# ### 4. Tutorial 4, 5, 6, 7
# ### 5. UCL dataset: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records#
