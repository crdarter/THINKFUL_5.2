# THINKFUL_5.2
Capstone code and write up

Problem: How to model a qualitative assessment, in this case the quality of white wine, from measureable chemical attributes.

Data: This dataset is public available for research. The details are described in [Cortez et al., 2009]. 

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems>, Elsevier, 47(4):547-553. ISSN: 0167-9236.

Process:

DATA PULL
The data is available via UCI's machine learning databases: http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

DATA CLEANING
Unlike typical csv, this data file uses a ";" as a delineator, and since the column headings in the original file use spaces in their names, both these issues will need to be accounted for.

DATA REVIEW
The data was reviewed in two phases:
1) Numpy mean, min and max summaries provided a quick snap shot of the range of values
2) The variables were visualized using histograms and box plots

This effort revealed that the sulfur dioxide variables were on a vastly different scale than the other variables and may prove problematic when using a linear regression technique if not re-scaled in some fashion.  It also confirmed that there were no missing values and that we had a sufficient number of cases for a linear regression analysis (n=4897).

REGRESSION TECHNIQUES
Ordinary Least Squares (OLS) Regression
The variables were converted to matrices to accommodate the OLS stats modeling syntax

When all variables are included the R-squared (the extent to which the model explains the variance in the data) is about 28%, which is not a horrible result, but leaves quite a bit of room for improvement, so additional regression techniques and model adjustments were explored.

Removing the variables with statistically insignificant coefficients, improved the adjusted R square only slightly.  Since one of the sulfur dioxide variables was dominating the model from a coefficient standpoint, the next step was to standardize the x variables.

OLS Regression (standardized x values)
The result of the standardized x values on the OLS model was to tame the coefficients and to reduce the conditional number (which was a warning given with  the original run).  As with the original regression effort, I sought to improve the model performance by removing variables from the model that did not have significant coefficients. Not surprisingly, they were the same variables that were removed in the non-standardized model.  And as before, this slightly improved the adjusted R square but still had a significantly reduced conditional number than the original variable models.

OLS Regression (factor analysis with standardized x values)
The other concern with the variables included, such as fixed_acidity and volatile_acidity, was that there was a strong potential for variables to be correlated, thereby creating a model with multiple independent variables expressing the same overarching characteristic.  Principal Component Analysis (PCA) was conducted on the standardized x values in an iterative fashion to see if an improved model could be achieved.    

From 2 factors to 6 factors the model's R squared performance went from 9.4% - 16.3%, it is even reasonable to suspect that the model would continue to improve if more factors were introduced; however, given that there are only 11 variables, going beyond 6 components, defeats the purpose of PCA>

Binary Logistic Regression 
Since the goal of the analysis is to determine quality from chemical attributes, it seemed reasonable to devise a dummy variable of the scaler quality metric into good white wine and poor white wine.  The range of values included in the data set was from 3-9 with an average of 5.9.  This suggests that 6 and above would be above average quality, and given that 67% of the cases qualify as good quality, this seemed like a reasonable threshold.  Changing the threshold to 5 and above, for example, would change the proportion of cases that qualify as good quality to +90%, which would not afford sufficient cases on poor quality for the binary regression model to be effective.

To prepare the data file for a binary logistic regression, the quality variable needs to be turned into a binary variable and a constant variable needs to be added.  The quality variable is achieved using lambda loop and the constant is simply =1 for each case.  The original file and these additional variables are written to a new csv file which the binary logit models will be run.

The initial binary logit model is run with all the variables and underperforms against the 28% R-squared, the strongest model achieved thus far.  Note: with binary logistic regression, we are evaluating p- R square or a pseudo R-squared.

Variables with statistically insignificant coefficients are removed, which do not alter the lackluster results significantly.

One other variation is attempted before abandoning the binary model altogether, which is to remove all variables that do not hold a strong coefficient.  This diminishes performance.

FINAL MODEL
The regression model that created the best outcome was the OLS regression model using standardized x values on a sub-set of the variables:  fixed_acidity,volatile_acidity,residual_sugar,free_sulfur_dioxide,density,pH,sulphates,alcohol])

The adjusted R-squared, is only slightly better than the full set of standardized variables and exactly the same to non-standardized sub-set of x values, but as indicated earlier, the conditional number is much improved and using the standardized values allows us to rank the predictive importance, like so:

Density (negative)
Residual sugar (positive)
Alcohol (positive)
Volatile acidity (negative)
pH (positive)
Sulphates (positive)
Fixed acidity (positive)
Free sulfur dioxide (positive)

CONCLUSION:
Quality white wines are more apt to be light and sweet.
