# Predicting and Understanding Major Power Outages in the United States
Project for Dsc 80 at UCSD

By Haihan Wang

# Introduction
Major power outages affect millions of people in the United States every year, disrupting homes, businesses, transportation, communication, and public safety. Understanding why outages happen and how long they last is essential for improving grid reliability and emergency preparedness.

The dataset provides detailed information about large-scale power outages in the United States from January 2000 to July 2016, along with contextual attributes describing the geographic, climatic, economic, and energy-use characteristics of the affected regions. These additional variables make it possible to examine not just when and where outages occur, but also the environmental and infrastructural conditions surrounding each event.

My analysis begins with data cleaning and exploratory data analysis to develop an understanding of the dataset and the relationships among its key variables. I also evaluate patterns of missingness to determine whether certain features are systematically absent based on outage characteristics.

The main research question guiding this project is: What factors contribute most to the duration and severity of major power outages? To explore this, I build statistical and machine-learning models aimed at predicting outage duration based on event characteristics and contextual information. This type of prediction is valuable because accurate insights into outage behavior can help utility companies better prepare for severe events—for example, by strengthening infrastructure in regions prone to weather-related outages or improving security measures in areas vulnerable to intentional disruptions.

The original raw dataset contains 1534 rows, each representing a unique outage, and 57 columns. While the dataset is rich, my project focuses on a selected subset of variables most relevant to outage timing, cause, geographic context, and customer impact:


And here are descriptions for important columns that we will use later. 

|Column                |Description|
|---                |---        |
|`'YEAR'`                |Year an outage occurred|
|`'MONTH'`                |Month an outage occurred|
|`'U.S._STATE'`                |State the outage occurred in|
|`'NERC.REGION'`                |North American Electric Reliability Corporation (NERC) regions involved in the outage event|
|`'CLIMATE.REGION'`                |U.S. Climate regions as specified by National Centers for Environmental Information (9 Regions)|
|`'ANOMALY.LEVEL'`                |Oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season|
|`'OUTAGE.START.DATE'`                |Day of the year when the outage event started|
|`'OUTAGE.START.TIME'`                |Time of the day when the outage event started|
|`'OUTAGE.RESTORATION.DATE'`                |Day of the year when power was restored to all the customers|
|`'OUTAGE.RESTORATION.TIME'`                |Time of the day when power was restored to all the customers|
|`'CAUSE.CATEGORY'`                |Categories of all the events causing the major power outages|
|`'OUTAGE.DURATION'`                |Duration of outage events (in minutes)|
|`'DEMAND.LOSS.MW'`                |Amount of peak demand lost during an outage event (in Megawatt) [but in many cases, total demand is reported]|
|`'CUSTOMERS.AFFECTED'`                |Number of customers affected by the power outage event|
|`'TOTAL.PRICE'`                |Average monthly electricity price in the U.S. state (cents/kilowatt-hour)|
|`'TOTAL.SALES'`                |Total electricity consumption in the U.S. state (megawatt-hour)|
|`'TOTAL.CUSTOMERS'`                |Annual number of total customers served in the U.S. state|
|`'POPPCT_URBAN'`                |Percentage of the total population of the U.S. state represented by the urban population (in %)|
|`'POPDEN_URBAN'`                |Population density of the urban areas (persons per square mile)|
|`'AREAPCT_URBAN'`                |Percentage of the land area of the U.S. state represented by the land area of the urban areas (in %)|


# Data Cleaning and Exploratory Data Analysis
The first step is to clean the data to make sure it is suitable for effective analysis. 
## Cleaning
1. I start by cutting the unnecessary rows and columns first and set the "OBS" column as the index. 

2. Next, outage start and restoration times were originally split into separate date and time fields. Since each pair represents a single point in time, I combined them to form two new timestamp columns:
	•	OUTAGE.START
	•	OUTAGE.RESTORATION

This better reflects the true timing of outage events and allows accurate calculation of outage duration when needed.


3. Several columns—particularly OUTAGE.DURATION and CUSTOMERS.AFFECTED—contained values equal to 0. Because major outages cannot have a duration of 0 minutes or affect 0 customers, these zeros indicate missing or unreported values, not real measurements.To prevent incorrect interpretations in modeling or summary statistics, I replaced 0s in these columns with np.nan. This preserves the integrity of the data-generating process by distinguishing true measurements from missing information.

4. 	After completing these steps, the resulting cleaned DataFrame forms the foundation for subsequent EDA, modeling, and fairness analysis.

Below is a preview of the cleaned dataset (subset of columns displayed):




|   OBS |   YEAR |   MONTH | U.S._STATE   | POSTAL.CODE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.START.DATE         | OUTAGE.START.TIME   | OUTAGE.RESTORATION.DATE    | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   HURRICANE.NAMES |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND | OUTAGE.START        | OUTAGE.RESTORATION   |
|------:|-------:|--------:|:-------------|:--------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------------|:--------------------|:---------------------------|:--------------------------|:-------------------|:------------------------|------------------:|------------------:|-----------------:|---------------------:|------------:|------------:|------------:|--------------:|------------:|------------:|------------:|--------------:|-------------:|-------------:|-------------:|----------------:|----------------:|----------------:|------------------:|---------------:|---------------:|---------------:|-------------------:|-----------------:|-----------------:|--------------------:|---------------:|----------------:|--------------:|----------------:|-------------:|---------------:|------------:|---------------:|------------:|---------------:|----------------:|-------------:|-----------:|----------------:|-------------------:|:--------------------|:---------------------|
|     1 |   2011 |       7 | Minnesota    | MN            | MRO           | East North Central |            -0.3 | normal             | Friday, July 1, 2011      | 5:00:00 PM          | Sunday, July 3, 2011       | 8:00:00 PM                | severe weather     | nan                     |               nan |              3060 |              nan |                70000 |       11.6  |        9.18 |        6.81 |          9.28 |     2332915 |     2114774 |     2113291 |       6562520 |      35.5491 |      32.225  |      32.2024 |         2308736 |          276286 |           10673 |           2595696 |        88.9448 |        10.644  |         0.4112 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |      5348119 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2011-07-01 17:00:00 | 2011-07-03 20:00:00  |
|     2 |   2014 |       5 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | Sunday, May 11, 2014      | 6:38:00 PM          | Sunday, May 11, 2014       | 6:39:00 PM                | intentional attack | vandalism               |               nan |                 1 |              nan |                  nan |       12.12 |        9.71 |        6.49 |          9.28 |     1586986 |     1807756 |     1887927 |       5284231 |      30.0325 |      34.2104 |      35.7276 |         2345860 |          284978 |            9898 |           2640737 |        88.8335 |        10.7916 |         0.3748 |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |      5457125 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2014-05-11 18:38:00 | 2014-05-11 18:39:00  |
|     3 |   2010 |      10 | Minnesota    | MN            | MRO           | East North Central |            -1.5 | cold               | Tuesday, October 26, 2010 | 8:00:00 PM          | Thursday, October 28, 2010 | 10:00:00 PM               | severe weather     | heavy wind              |               nan |              3000 |              nan |                70000 |       10.87 |        8.19 |        6.07 |          8.15 |     1467293 |     1801683 |     1951295 |       5222116 |      28.0977 |      34.501  |      37.366  |         2300291 |          276463 |           10150 |           2586905 |        88.9206 |        10.687  |         0.3924 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |      5310903 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2010-10-26 20:00:00 | 2010-10-28 22:00:00  |
|     4 |   2012 |       6 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | Tuesday, June 19, 2012    | 4:30:00 AM          | Wednesday, June 20, 2012   | 11:00:00 PM               | severe weather     | thunderstorm            |               nan |              2550 |              nan |                68200 |       11.79 |        9.25 |        6.71 |          9.19 |     1851519 |     1941174 |     1993026 |       5787064 |      31.9941 |      33.5433 |      34.4393 |         2317336 |          278466 |           11010 |           2606813 |        88.8954 |        10.6822 |         0.4224 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |      5380443 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2012-06-19 04:30:00 | 2012-06-20 23:00:00  |
|     5 |   2015 |       7 | Minnesota    | MN            | MRO           | East North Central |             1.2 | warm               | Saturday, July 18, 2015   | 2:00:00 AM          | Sunday, July 19, 2015      | 7:00:00 AM                | severe weather     | nan                     |               nan |              1740 |              250 |               250000 |       13.07 |       10.16 |        7.74 |         10.43 |     2028875 |     2161612 |     1777937 |       5970339 |      33.9826 |      36.2059 |      29.7795 |         2374674 |          289044 |            9812 |           2673531 |        88.8216 |        10.8113 |         0.367  |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |      5489594 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2015-07-18 02:00:00 | 2015-07-19 07:00:00  |



## Exploratory Data Analysis

### Univariate Analysis
Univariate Analysis

Distribution of Outage Duration

<iframe
  src="assets/outage-duration-hist.html"
  width="800"
  height="600"
  frameborder="0">
</iframe>

Interpretation:
The distribution of outage durations is strongly right-skewed, meaning that while most outages last only a few hours, a smaller number of events extend for several days. This long tail suggests that predicting outage duration is challenging, as a minority of extreme events exert substantial influence on the overall distribution. These insights motivated the use of log transforms and tree-based models in later modeling steps.

Distribution of Outages Across Climate Regions

<iframe
  src="assets/climate-region-bar.html"
  width="850"
  height="600"
  frameborder="0">
</iframe>

Interpretation:
This bar chart shows the distribution of outages across different climate regions in the United States. Some regions, particularly those prone to severe weather patterns, appear more frequently in the dataset.



### Bivariate Analysis

I conducted several bivariate analyses, and one of the most meaningful relationships appeared between outage duration and the cause category. Different causes lead to outages with very different average durations, suggesting that some types of events are inherently harder to restore from than others. 

Average Outage Duration Across Different Cause Categories
<iframe
  src="assets/cause-duration-bar.html"
  width="850"
  height="600"
  frameborder="0">
</iframe>
This chart highlights that fuel supply emergency causes the longest outages on average, while equipment-related or operability-related failures tend to be resolved more quickly.



### Grouping and Aggregates
To uncover broader trends in outage behavior, I examined how outage duration varies across different combinations of climate characteristics. The pivot table below summarizes the average outage duration grouped by both Climate Region and Climate Category, providing a two-dimensional view of how geography and weather patterns interact to influence outage severity.

| CLIMATE.REGION     |     cold |    normal |    warm |
|:-------------------|---------:|----------:|--------:|
| Central            | 2676.34  | 2682.15   | 2080.9  |
| East North Central | 6568.79  | 5207.71   | 3022.12 |
| Northeast          | 3568.77  | 2261.33   | 3990.31 |
| Northwest          |  874.681 |  733.612  | 2212.56 |
| South              | 1977.4   | 3685.44   | 1672.1  |
| Southeast          | 1707.07  | 2392.27   | 2528.94 |
| Southwest          |  499.208 |  283.261  | 5127.68 |
| West               | 1735.17  | 1142.32   | 1942.02 |
| West North Central |  200     |   28.4286 | 2486.5  |



# Assessment of Missingness

## NMAR Analysis
One column in my dataset that is likely NMAR (Not Missing At Random) is DEMAND.LOSS.MW. Many outages contain information about customers affected and outage duration but are missing demand loss. This suggests that the missingness may be related to the value itself—utility companies might fail to report demand loss when the value is unknown, difficult to estimate, or unusually small or large. 

To investigate missingness further, I performed permutation tests to determine whether the missingness of DEMAND.LOSS.MW depends on other variables in the dataset. Below I present one example where missingness is not dependent and one where it is dependent.

## Missingness Dependency
### Test 1: Does missingness depend on OUTAGE.DURATION? (It does NOT)

**Null Hypothesis (H₀):**
The average outage duration is the same for outages with missing DEMAND.LOSS.MW and for those with observed values. Missingness is independent of outage duration.

**Alternative Hypothesis (H₁):**
The average outage duration differs between outages with missing and non-missing demand loss. Missingness depends on outage duration.

I used the difference in means as the test statistic.
**Cause Category by Missingness of DEMAND.LOSS.MW**

<iframe
  src="assets/missing-cause-proportion.html"
  width="850"
  height="600"
  frameborder="0"
></iframe>


<iframe
  src="assets/missing-duration-permutation.html"
  width="800"
  height="600"
  frameborder="0">
</iframe>

Results:
	•	Observed difference in means: ≈ 80 minutes
	•	P-value: 0.815

Since the p-value is far above 0.05, I fail to reject the null hypothesis.
This suggests that missingness of DEMAND.LOSS.MW does not depend on outage duration.



### Test 2: Does missingness depend on NERC.REGION? (It DOES)

**Null Hypothesis (H₀):**
The distribution of NERC regions is the same for outages with missing and non-missing DEMAND.LOSS.MW. Missingness is independent of NERC region.

**Alternative Hypothesis (H₁):**
The distribution of regions differs for missing vs. non-missing cases. Missingness depends on NERC region.

I used Total Variation Distance (TVD) as the test statistic.
**NERC Region by Missingness of DEMAND.LOSS.MW**

<iframe
  src="assets/missing-region-proportion.html"
  width="850"
  height="600"
  frameborder="0"
></iframe>


<iframe
  src="assets/missing-region-permutation.html"
  width="800"
  height="600"
  frameborder="0">
</iframe>

Results:
	•	Observed TVD: 0.192
	•	P-value: 0.0

Since the p-value is effectively zero, I reject the null hypothesis.
This shows that missingness of DEMAND.LOSS.MW depends strongly on the NERC region, suggesting regional differences in reporting practices or measurement capabilities.

### Conclusion

Overall, DEMAND.LOSS.MW appears to be NMAR, and its missingness is not explained by outage duration, but does depend on the NERC region. This indicates that some regions systematically fail to report demand-loss information, which may reflect differences in data collection infrastructure or regulatory standards.




# Hypothesis Testing

I want to find out the characteristics that influence how long the outage lasts. To do this, I will compare the outage durations among different categories for many columns (for this test, I use Severe Weather vs Equipment Failure in the "CAUSE.CATEGORY" column).

**Null Hypothesis (H₀):**
The average outage duration is the same for outages caused by the severe weather and equipment failure.

**Alternative Hypothesis (H₁):**
The average outage duration differs for outages caused by the severe weather and equipment failure.

**Test Statistic:** I will use the absolute difference in mean outage duration between the two groups¶


I performed a permutation test with 20,000 simulations to generate an empirical distribution of the test statisic. Significance level I used was 0.05

Result:
Observed Difference in Means: 2121.7673656618613
P-value: 01035

With the result, we reject the null hypothesis because the results are statistically significant. We conclude that the average outage duration differs for outages caused by the severe weather and equipment failure.

The plot below shows the observed difference against the empirical distribution of differences from the permutation tests.
<iframe
  src="assets/hypothesis-permutation.html"
  width="850"
  height="600"
  frameborder="0"
></iframe>



# Framing a Prediction Problem

The goal of my prediction task is to predict the severity of a major power outage, measured by its duration in minutes. Because the response variable is numeric and continuous, this is a regression problem rather than a classification problem.

**Response Variable:**
	•	OUTAGE.DURATION — the total length of time customers were without power
I chose this variable because outage duration is one of the most important indicators of outage severity.

**Evaluation Metric:**
I evaluate the performance of my model using Mean Squared Error (MSE).
I chose MSE because it penalizes large errors more heavily, which is important because severely misestimating long outages is more harmful than small mistakes on short outages. Unlike MAE, it provides smoother gradients, which benefits most regression algorithms.

**Features Available at the Time of Prediction:**
	•	CUSTOMERS.AFFECTED (initial reported affected customers)
	•	CAUSE.CATEGORY (cause classification is typically known early from reports or diagnostics)
	•	U.S._STATE, NERC.REGION, CLIMATE.REGION (location-based features)
	•	YEAR, MONTH (temporal context)
	•	TOTAL.SALES, TOTAL.PRICE, TOTAL.CUSTOMERS
	•	Urbanization-related variables such as POPPCT_URBAN, POPDEN_URBAN, AREAPCT_URBAN
    •   And all other features in the original dataframe (since I didn't take any remaining columns off)



# Baseline Model
My baseline model is a regression model that predicts OUTAGE.DURATION using two features known at the time an outage begins: CUSTOMERS.AFFECTED and CAUSE.CATEGORY. I chose these features because the initial scale of customer impact often reflects outage severity, and the underlying cause can strongly influence how long repairs take.

Feature Types and Encodings
	•	Quantitative: CUSTOMERS.AFFECTED — used directly as a numeric predictor.
	•	Nominal: CAUSE.CATEGORY — one-hot encoded because it is a categorical, non-ordered variable.
	•	Ordinal: none in this model.

These transformations are handled using a ColumnTransformer, and preprocessing is combined with a Linear Regression estimator inside an sklearn Pipeline for reliable and consistent training.

Performance

Evaluated with Mean Squared Error (MSE) using an 80/20 train–test split:
	•	Train MSE: 18,877,135
	•	Test MSE: 14,608,807

Assessment

The model provides a reasonable benchmark but is not highly accurate, as the large MSE values suggest. With only two features and a linear form, the model likely underfits the complexity of outage duration. This indicates that adding engineered features and tuning hyperparameters will be necessary for a more predictive final model.

# Final Model
To improve upon the baseline model, I incorporated additional features and applied several feature-engineering transformations based on how outages occur in the real world. The final model uses the following predictors:
	•	RES.PRICE (quantitative): Electricity price reflects regional infrastructure costs and economic conditions, which can influence outage severity and restoration difficulty.
	•	COM.CUST.PCT (quantitative): Outages affecting a high proportion of commercial customers may take longer to restore due to economic priority or greater complexity.
	•	NERC.REGION (nominal): Different NERC regions have different infrastructure, climate exposure, and grid reliability standards, all of which influence outage duration.
	•	CUSTOMERS.AFFECTED (quantitative): Larger outages typically require more time and resources to resolve.
	•	CAUSE.CATEGORY (nominal): Causes such as weather or equipment failure differ significantly in repair difficulty.

These added features are appropriate because they capture economic conditions, geographic variation, and outage complexity—factors that directly influence how long it takes to restore power.

Feature Engineering

I created three engineered feature groups:
	•	Log-scaled feature: CUSTOMERS.AFFECTED — reduces skew caused by extremely large outages.
	•	Quantile-normalized feature: RES.PRICE — matches the assumption of many models that predictors follow a smoother distribution.
	•	Standard-scaled feature: COM.CUST.PCT — ensures the variable is on a consistent scale before entering the model.
	•	One-hot encoding: applied to CAUSE.CATEGORY and NERC.REGION.

These engineered features were selected prior to modeling because they help address skew, normalize distributions, and properly encode categorical structure—improving model stability and predictive ability.

Model and Hyperparameter Tuning

I used a Random Forest Regressor, which is well-suited for capturing nonlinear relationships present in outage behavior.

To select the best model, I used GridSearchCV with 5-fold cross-validation over the following hyperparameters:
	•	n_estimators: [100, 300]
	•	max_depth: [None, 5, 10, 20]
	•	min_samples_leaf: [1, 5]

The best combination was:

{'model__max_depth': 20, 'model__min_samples_leaf': 5, 'model__n_estimators': 300}

This reflects a moderately deep forest with many trees and a leaf size that reduces overfitting.

Performance Comparison

Model	        Train MSE	    Test MSE
Baseline:	    18,877,135	;    14,608,807
Final Model:	    10,261,869	 ;   12,576,249

The final model substantially improved performance:
	•	Test MSE decreased by ~2 million compared to baseline.
	•	Train MSE dropped even more significantly, suggesting the expanded feature set captures meaningful structure in outage duration.

Assessment

The final model is clearly an improvement over the baseline. The added features reflect realistic, data-generating factors (regional variation, customer demographics, infrastructure characteristics) and meaningfully improve predictive accuracy. The Random Forest model captures nonlinear patterns that a linear baseline model could not.



# Fairness Analysis
To evaluate whether my final regression model behaves fairly across different types of outages, I examined its performance for two outage cause categories that appear frequently in the test set:
	•	Group X: Severe Weather
	•	Group Y: Intentional Attack

These groups differ meaningfully in how outages unfold. Severe weather outages are typically widespread and unpredictable, while intentional attacks tend to be localized events. If the model systematically performs worse for one group, this indicates unequal treatment that could affect planning or resource allocation.

Evaluation Metric: RMSE

Because this is a regression model, I use Root Mean Squared Error (RMSE) to measure prediction performance. A fair model should achieve similar RMSE values for both groups.

**Hypotheses**
	•	Null Hypothesis (H₀):
The model is fair. The RMSE for severe weather and intentional attack outages is the same; any difference is due to random chance.
	•	Alternative Hypothesis (H₁):
The model is unfair. The RMSE differs between the two groups, meaning the model performs worse for one of them.

Test Statistic & Significance Level

The test statistic is the difference in RMSE:

RMSE_X - RMSE_Y

I use a two-sided permutation test with 2000 permutations and a significance level of α = 0.05.

**Results**
	•	Observed RMSE difference:

RMSE_X - RMSE_Y = 3533.26

The model makes much larger errors on severe weather outages than intentional attacks.
	•	Permutation Test Result:
p-value = 0.002

Since the p-value is well below 0.05, we reject the null hypothesis.

Conclusion

There is strong statistical evidence that the model’s prediction accuracy is not equal across outage causes. The model performs significantly worse for outages caused by severe weather, indicating a fairness concern.

This suggests that the model may struggle to capture the greater variability and complexity of weather-related outages. Future modeling improvements may require incorporating additional climate or infrastructure features to reduce this disparity.


