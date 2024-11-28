#Data Loading and Exploration
#Loading the Dataset
import pandas as pd

# Load the dataset
url = 'hotel_bookings.csv'  # Replace with actual file path or Kaggle API access
data = pd.read_csv(url)

# Display the first few rows
print(data.head())

#Initial Exploration
# Get basic information about the dataset
data.info()

# Get descriptive statistics for numerical columns
print(data.describe())

# Check for missing values
print(data.isnull().sum())

#Data Cleaning
#Handling Missing Values
# Fill missing values or drop columns/rows with excessive missing values
data.fillna(method='ffill', inplace=True)  # Forward fill for missing values
# Alternatively, use dropna() to remove rows with missing values
data=data.dropna()
print(data.isnull().sum())

#Exploratory Data Analysis (EDA)
#Univariate Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the distribution of lead_time
plt.figure(figsize=(10, 6))
sns.histplot(data['lead_time'], kde=True, color='blue')
plt.title('Distribution of Lead Time')
plt.xlabel('Lead Time (days)')
plt.ylabel('Frequency')
plt.show()

# Visualizing the distribution of adr (Average Daily Rate)
plt.figure(figsize=(10, 6))
sns.histplot(data['adr'], kde=True, color='green')
plt.title('Distribution of Average Daily Rate (ADR)')
plt.xlabel('Average Daily Rate (ADR)')
plt.ylabel('Frequency')
plt.show()

#Bivariate Analysis
# Visualizing the cancellation rate with respect to lead time
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_canceled', y='lead_time', data=data)
plt.title('Cancellation Rate vs Lead Time')
plt.xlabel('Cancellation Status')
plt.ylabel('Lead Time (days)')
plt.show()

# Visualizing ADR with respect to cancellation
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_canceled', y='adr', data=data)
plt.title('Cancellation Rate vs ADR')
plt.xlabel('Cancellation Status')
plt.ylabel('Average Daily Rate (ADR)')
plt.show()

#Statistical Analysis
#Hypothesis Testing
from scipy import stats

# T-test to compare lead_time between canceled and non-canceled bookings
canceled = data[data['is_canceled'] == 1]['lead_time']
not_canceled = data[data['is_canceled'] == 0]['lead_time']

t_stat, p_value = stats.ttest_ind(canceled, not_canceled)
print(f'T-statistic: {t_stat}, P-value: {p_value}')

#Segmentation
# Create customer segments based on the number of guests
data['customer_segment'] = pd.cut(data['adults'] + data['children'] + data['babies'], bins=[0, 1, 2, 3, 5, 10, 20], labels=["1", "2", "3", "4-5", "6-10", "10+"])

# Group by customer segments and calculate cancellation rates
segment_cancellation = data.groupby('customer_segment')['is_canceled'].mean()
print(segment_cancellation)

#Insights and Reporting
#bar plot showing cancellation rates for different customer segments
# Visualize cancellation rates by customer segment
plt.figure(figsize=(10, 6))
sns.barplot(x=segment_cancellation.index, y=segment_cancellation.values, palette='viridis')
plt.title('Cancellation Rate by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Cancellation Rate')
plt.show()

#Lead Time Analysis
#Analyze the relationship between lead_time and is_canceled.
#Compare adr (average daily rate) across different lead_time bins.

# Binning lead_time for better visualization
data['lead_time_bins'] = pd.cut(data['lead_time'], bins=[0, 30, 90, 180, 365, 1000], 
                                labels=['0-30 days', '31-90 days', '91-180 days', '181-365 days', '1+ years'])

# Plotting cancellation rates by lead_time bins
plt.figure(figsize=(12, 6))
cancel_rate = data.groupby('lead_time_bins')['is_canceled'].mean()
sns.barplot(x=cancel_rate.index, y=cancel_rate.values, palette='coolwarm')
plt.title('Cancellation Rate by Lead Time')
plt.xlabel('Lead Time (Bins)')
plt.ylabel('Cancellation Rate')
plt.show()

# Plotting ADR by lead_time bins
plt.figure(figsize=(12, 6))
adr_by_lead_time = data.groupby('lead_time_bins')['adr'].mean()
sns.barplot(x=adr_by_lead_time.index, y=adr_by_lead_time.values, palette='viridis')
plt.title('Average Daily Rate (ADR) by Lead Time')
plt.xlabel('Lead Time (Bins)')
plt.ylabel('Average Daily Rate (ADR)')
plt.show()

#Booking Patterns by Customer Type
#Analyze is_canceled and adr for repeated vs. non-repeated guests.
# Cancellation rate and ADR comparison for repeated vs. non-repeated guests
repeat_guest_stats = data.groupby('is_repeated_guest')[['is_canceled', 'adr']].mean()
repeat_guest_stats = repeat_guest_stats.rename(columns={'is_canceled': 'Cancellation Rate', 'adr': 'Average ADR'})

# Visualize the stats
plt.figure(figsize=(12, 6))
repeat_guest_stats.plot(kind='bar', figsize=(12, 6), color=['#1f77b4', '#ff7f0e'])
plt.title('Booking Patterns by Customer Type')
plt.ylabel('Rate / ADR')
plt.xlabel('Customer Type (0: New, 1: Repeat)')
plt.xticks(ticks=[0, 1], labels=['New Guests', 'Repeat Guests'], rotation=0)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Customer segmentation
#Segment customers based on the total number of guests (adults, children, and babies).
data['customer_segment'] = pd.cut(data['adults'] + data['children'] + data['babies'], 
                                  bins=[0, 1, 2, 4, 6, 10, 20], 
                                  labels=["1", "2", "3-4", "5-6", "7-10", "10+"])

# Calculate cancellation rates and ADR by segment
segment_stats = data.groupby('customer_segment')[['is_canceled', 'adr']].mean()
segment_stats = segment_stats.rename(columns={'is_canceled': 'Cancellation Rate', 'adr': 'Average ADR'})

# Visualize the stats
plt.figure(figsize=(12, 6))
segment_stats.plot(kind='bar', figsize=(12, 6), color=['#d62728', '#2ca02c'])
plt.title('Cancellation Rate and ADR by Customer Segment')
plt.ylabel('Rate / ADR')
plt.xlabel('Customer Segment')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

