#CAPSTONE

#Data source citation:
#  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 

#  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#  Modeling wine preferences by data mining from physicochemical properties.
#  In Decision Support Systems>, Elsevier, 47(4):547-553. ISSN: 0167-9236.

#  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
#                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
#                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

import pandas as pd
import numpy as np

df = pd.read_csv(
    filepath_or_buffer='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    skiprows=1, sep=';')

df.columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
	
print df.info()

print np.mean(df)
print np.min(df)
print np.max(df)

#CREATING HISTOGRAMS
import matplotlib.pyplot as plt
plt.figure()
df.hist(column='fixed_acidity')
plt.savefig('fixed_acidity_histogram.png')

plt.figure()
df.hist(column='volatile_acidity')
plt.savefig('volatile_acidity_histogram.png')

plt.figure()
df.hist(column='citric_acid')
plt.savefig('citric_acid_histogram.png')

plt.figure()
df.hist(column='residual_sugar')
plt.savefig('residual_sugar_histogram.png')

plt.figure()
df.hist(column='chlorides')
plt.savefig('chlorides_histogram.png')

plt.figure()
df.hist(column='free_sulfur_dioxide')
plt.savefig('free_sulfur_dioxide_histogram.png')

plt.figure()
df.hist(column='total_sulfur_dioxide')
plt.savefig('total_sulfur_dioxide_histogram.png')

plt.figure()
df.hist(column='density')
plt.savefig('density_histogram.png')

plt.figure()
df.hist(column='pH')
plt.savefig('pH_histogram.png')

plt.figure()
df.hist(column='sulphates')
plt.savefig('sulphates_histogram.png')

plt.figure()
df.hist(column='alcohol')
plt.savefig('alcohol_histogram.png')

plt.figure()
df.hist(column='quality')
plt.savefig('quality_histogram.png')

#CREATING BOXPLOTS
plt.figure()
df.boxplot(column='fixed_acidity')
plt.savefig('fixed_acidity_boxplot.png')

plt.figure()
df.boxplot(column='volatile_acidity')
plt.savefig('volatile_acidity_boxplot.png')

plt.figure()
df.boxplot(column='citric_acid')
plt.savefig('citric_acid_boxplot.png')

plt.figure()
df.boxplot(column='residual_sugar')
plt.savefig('residual_sugar_boxplot.png')

plt.figure()
df.boxplot(column='chlorides')
plt.savefig('chlorides_boxplot.png')

plt.figure()
df.boxplot(column='free_sulfur_dioxide')
plt.savefig('free_sulfur_dioxide_boxplot.png')

plt.figure()
df.boxplot(column='total_sulfur_dioxide')
plt.savefig('total_sulfur_dioxide_boxplot.png')

plt.figure()
df.boxplot(column='density')
plt.savefig('density_boxplot.png')

plt.figure()
df.boxplot(column='pH')
plt.savefig('pH_boxplot.png')

plt.figure()
df.boxplot(column='sulphates')
plt.savefig('sulphates_boxplot.png')

plt.figure()
df.boxplot(column='alcohol')
plt.savefig('alcohol_boxplot.png')

plt.figure()
df.boxplot(column='quality')
plt.savefig('quality_boxplot.png')

#Ordinary Least Squares (OLS) Regression Model
import statsmodels.api as sm

fixed_acidity = df['fixed_acidity']
volatile_acidity = df['volatile_acidity']
citric_acid = df['citric_acid']
residual_sugar = df['residual_sugar']
chlorides = df['chlorides']
free_sulfur_dioxide = df['free_sulfur_dioxide']
total_sulfur_dioxide = df['total_sulfur_dioxide']
density = df['density']
pH = df['pH']
sulphates = df['sulphates']
alcohol = df['alcohol']
quality = df['quality']


y = np.matrix(quality).transpose()
x1 = np.matrix(fixed_acidity).transpose()
x2 = np.matrix(volatile_acidity).transpose()
x3 = np.matrix(citric_acid).transpose()
x4 = np.matrix(residual_sugar).transpose()
x5 = np.matrix(chlorides).transpose()
x6 = np.matrix(free_sulfur_dioxide).transpose()
x7 = np.matrix(total_sulfur_dioxide).transpose()
x8 = np.matrix(density).transpose()
x9 = np.matrix(pH).transpose()
x10 = np.matrix(sulphates).transpose()
x11 = np.matrix(alcohol).transpose()

x = np.column_stack([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()

#OLS Regression Model (cont) - Revising the model to exclude variables that do not hold coefficients that are statistically significant
x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()

#OLS Regression Model (cont) - Since the model still falls flat, tried standardizing the values as some variables hold a wider range of values than others
from sklearn.preprocessing import StandardScaler
x = np.column_stack([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

X = sm.add_constant(X_std)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/Standardized x values'
print f.summary()

#OLS Regression Model (cont) - Using a sub-set of the standardized x values, limited to those with statistically significant coefficients
from sklearn.preprocessing import StandardScaler
x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

X = sm.add_constant(X_std)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/Standardized sub-set of x values'
print f.summary()


#OLS Regression Model (cont) - Since the model still falls flat, tried crafting factors (in addition to standardizing values) to reduce the impact of covariance
#OLS Regression Model - 2 Factors
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

X = sm.add_constant(Y_sklearn)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/2 Factors'
print f.summary()

#OLS Regression Model - 3 Factors
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

sklearn_pca = sklearnPCA(n_components=3)
Y_sklearn = sklearn_pca.fit_transform(X_std)

X = sm.add_constant(Y_sklearn)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/3 Factors'
print f.summary()

#OLS Regression Model - 4 Factors
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

sklearn_pca = sklearnPCA(n_components=4)
Y_sklearn = sklearn_pca.fit_transform(X_std)

X = sm.add_constant(Y_sklearn)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/4 Factors'
print f.summary()

#OLS Regression Model - 5 Factors
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

sklearn_pca = sklearnPCA(n_components=5)
Y_sklearn = sklearn_pca.fit_transform(X_std)

X = sm.add_constant(Y_sklearn)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/5 Factors'
print f.summary()

#OLS Regression Model - 6 Factors (with only 11 variables - using more than 6 factors reduces the value of factoring)
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

sklearn_pca = sklearnPCA(n_components=6)
Y_sklearn = sklearn_pca.fit_transform(X_std)

X = sm.add_constant(Y_sklearn)
model = sm.OLS(y,X)
f = model.fit()

print 'OLS Regression Model w/6 Factors'
print f.summary()


#Exploring Binary Logistic Regression as an alternative means of modeling white wine quality from measurable chemical components

#Preparing data for Binary Regression
df = pd.read_csv(
    filepath_or_buffer='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    skiprows=1, sep=';')

df.columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
df['constant'] = 1
quality_binary = df['quality'].map(lambda x: int(x >= 6))
df['quality_binary'] = quality_binary

df.to_csv('whitewine_clean.csv', header=True, index=False)

#Binary Logistic Regression

df = pd.read_csv('whitewine_clean.csv')
ind_vars = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'constant']

logit = sm.Logit(df['quality_binary'],df[ind_vars])
result = logit.fit()

coeff = result.params

print result.summary(logit)

#Binary Logistic Regression (cont) - Removing the variables with statistically insignificant coefficients
df = pd.read_csv('whitewine_clean.csv')  

ind_vars = ['volatile_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'constant']

logit = sm.Logit(df['quality_binary'],df[ind_vars])
result = logit.fit()

coeff = result.params

print result.summary(logit)

#Binary Logistic Regression (cont) - Further refining the model to incorporate just the strongest coefficients

df = pd.read_csv('whitewine_clean.csv')  

ind_vars = ['volatile_acidity', 'density', 'pH', 'sulphates','constant']

logit = sm.Logit(df['quality_binary'],df[ind_vars])
result = logit.fit()

coeff = result.params

print result.summary(logit)

print "Conclusion: the best model was achieved using...OLS Regression Modeling on standardized values for variables with statistically significant coefficients"
print 'Final Model'
from sklearn.preprocessing import StandardScaler
x = np.column_stack([x1,x2,x4,x6,x8,x9,x10,x11])
X_std = StandardScaler().fit_transform(x)

X = sm.add_constant(X_std)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()
