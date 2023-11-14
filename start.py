import pandas as pd
import csv

# TODO:
# 1. Descriptive statistics for initial dataset
# 2. Write k-means clustering algorithm adjustable for different #s of variables
# 3. K-means clustering generally
# 4. K-means clustering without demographics

# Considerations for later
# 1. Some columns have empty value, such as ES47 in the excel
# 2. Number of people with 0 siblings does not match the only children count


dataset = pd.read_csv("Data/responses.csv")
# print(dataset.describe())

# ignored --
#dataset_cleaned = dataset.dropna()
# print(dataset_cleaned.describe())

#--------------------------------------------------------------------------------------
#-----------PART 1: INITIAL DEMOGRAPHIC BREAKDOWN -------------------------------------
#--------------------------------------------------------------------------------------

# Age Breakdown
age_breakdown = dataset['Age'].value_counts()
print("Age Summary Statistics: " + age_breakdown.to_string())

# Height Breakdown
height_buckets = pd.cut(dataset['Height'], bins=range(
    145, 210, 5), include_lowest=True)
height_bucket_counts = height_buckets.value_counts(sort=False)
print("Height Bucket Counts:" + height_bucket_counts.to_string())

# Gender Breakdown
gender_breakdown = dataset['Gender'].value_counts()
print("Gender Summary Statistics:" + gender_breakdown.to_string())

# Weight
weight_buckets = pd.cut(dataset['Weight'], bins=range(40, 170, 5), include_lowest=True)
weight_bucket_counts = weight_buckets.value_counts(sort=False)
print("Weight Bucket Counts:" + weight_bucket_counts.to_string())

# Number of siblings
sibs_breakdown = dataset['Number of siblings'].value_counts()
print("Sibling # Value Counts:" + sibs_breakdown.to_string())

# Left - right handed
hand_breakdown = dataset['Left - right handed'].value_counts()
print("Left or Right Handed Value Counts:" + hand_breakdown.to_string())

# Education
ed_breakdown = dataset['Education'].value_counts()
print("Education Value Counts:" + ed_breakdown.to_string())

# Only child
onlychild_breakdown = dataset['Only child'].value_counts()
print("Only Child Value Counts:" + onlychild_breakdown.to_string())

# Village - town
rural_breakdown = dataset['Village - town'].value_counts()
print("Village or Town Value Counts:" + rural_breakdown.to_string())

# House - block of flats'
house_breakdown = dataset['House - block of flats'].value_counts()
print("House or Block of Flats Value Counts:" + house_breakdown.to_string())

#--------------------------------------------------------------------------------------
#-----------PART 2: K-MEANS CLUSTERING  -----------------------------------------------
#--------------------------------------------------------------------------------------

def lloyds_kmeans()