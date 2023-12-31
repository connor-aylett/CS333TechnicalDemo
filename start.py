import numpy as np
import pandas as pd
import csv
import math

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

# --------------------------------------------------------------------------------------
# -----------PART 1: INITIAL DEMOGRAPHIC BREAKDOWN -------------------------------------
# --------------------------------------------------------------------------------------

# Age Breakdown
age_breakdown = dataset['Age'].value_counts()
#print("Age Summary Statistics: " + age_breakdown.to_string())

# Height Breakdown
height_buckets = pd.cut(dataset['Height'], bins=range(
    145, 210, 5), include_lowest=True)
height_bucket_counts = height_buckets.value_counts(sort=False)
#print("Height Bucket Counts:" + height_bucket_counts.to_string())

# Gender Breakdown
gender_breakdown = dataset['Gender'].value_counts()
#print("Gender Summary Statistics:" + gender_breakdown.to_string())

# Weight
weight_buckets = pd.cut(dataset['Weight'], bins=range(
    40, 170, 5), include_lowest=True)
weight_bucket_counts = weight_buckets.value_counts(sort=False)
#print("Weight Bucket Counts:" + weight_bucket_counts.to_string())

# Number of siblings
sibs_breakdown = dataset['Number of siblings'].value_counts()
# print("Sibling # Value Counts:" + sibs_breakdown.to_string())

# Left - right handed
hand_breakdown = dataset['Left - right handed'].value_counts()
#print("Left or Right Handed Value Counts:" + hand_breakdown.to_string())

# Education
ed_breakdown = dataset['Education'].value_counts()
#print("Education Value Counts:" + ed_breakdown.to_string())

# Only child
onlychild_breakdown = dataset['Only child'].value_counts()
#print("Only Child Value Counts:" + onlychild_breakdown.to_string())

# Village - town
rural_breakdown = dataset['Village - town'].value_counts()
#print("Village or Town Value Counts:" + rural_breakdown.to_string())

# House - block of flats'
house_breakdown = dataset['House - block of flats'].value_counts()
#print("House or Block of Flats Value Counts:" + house_breakdown.to_string())

# Mapping of categorical columns to 1-5 number scales
smoking_mapping = {'never smoked': 1, 'tried smoking': 2, 'former smoker': 4, 'current smoker': 5}

alcohol_mapping = {'never': 1, 'social drinker': 3, 'drink a lot': 5}

punctuality_mapping = {'i am always on time': 1, 'i am often early': 3, 'i am often running late': 5}

lying_mapping = {'never': 1, 'only to avoid hurting someone': 3, 'sometimes': 4, 'everytime it suits me': 5}

internet_mapping = {'less than an hour a day': 1, 'few hours a day': 3, 'most of the day': 5}

gender_mapping = {'female': 0, 'male': 1}

handed_mapping = {'left handed': 0, 'right handed': 1}

education_mapping = {'primary school': 1, 'secondary school': 2, 'college/bachelor degree': 4, 'masters degree': 5}

onlychild_mapping = {'no': 0, 'yes': 1}

city_mapping = {'village': 0, 'city': 1}

house_mapping = {'block of flats': 0, 'house/bungalow': 1}


# --------------------------------------------------------------------------------------
# -----------PART 2: K-MEANS CLUSTERING  -----------------------------------------------
# --------------------------------------------------------------------------------------

# data = dataset, k = # of centroids, iterations is maximum (won't reach if converge)
def lloyds_kmeans(data, k, iterations):
    # randomly initialize centroids

    centroids = data.sample(n=k).values
    clusters = [[]*k]

    for i in range(iterations):
        print("Starting iteration " + str(i))

        # calculate distance from each point to all centroids
        print(len(data))
        print(range(k))
        distances = [[0]*len(data) for x in range(k)]
        print("Length of distances: ", len(distances))
        print("Vertical length of distances", len(distances[0]))
        for centroidindex in range(k):
            print("Centroid index: " + str(centroidindex))
            currentcentroid = centroids[i]
            for rowindex in range(len(data)):
                print("Row index: " + str(rowindex))
                currentpoint = data.iloc[rowindex]
                distances[centroidindex][rowindex] = distance(
                    currentcentroid, currentpoint)

        # assign point to correct cluster
        for rowindex in range(len(data)):
            tempmin = 0
            for centroidindex in range(k):
                if distances[centroidindex][rowindex] < distances[tempmin][rowindex]:
                    tempmin = centroidindex
            clusters[tempmin].append(data.iloc[rowindex])

        # new centroids based on mean of each cluster -- CONNOR
        for clusterindex in range(k):
            centroidsnew = meandata(clusters[clusterindex])

        # check if new centroids are same as old ones (convergence)
        if centroids.equals(centroidsnew):
            print("Converged after " + i + " iterations")
            break
        centroids = centroidsnew

    # calculate distance from each point to final centroids
    distances = [[]*len(data) for x in range[k]]
    for centroidindex in range(k):
        currentcentroid = centroids[i]
        for rowindex in range(len(data)):
            currentpoint = data.iloc[rowindex]
            distances[centroidindex][rowindex] = distance(
                currentcentroid, currentpoint)

    # assign point to final cluster
    for rowindex in range(len(data)):
        tempmin = 0
        for centroidindex in range(k):
            if distances[centroidindex][rowindex] < distances[tempmin][rowindex]:
                tempmin = centroidindex
        clusters[tempmin].append(data.iloc[rowindex])

    return centroids, clusters


def distance(user1, user2):
    distance = 0

    for x in range(len(user1)):
        temp = 0
        if isinstance(user1[x], np.float64):
            temp += user1[x] - user2[x]
        elif isinstance(user1[x], np.int64):
            temp += user1[x] - user2[x]
        elif user1[x] in smoking_mapping and user2[x] in smoking_mapping:
            temp += smoking_mapping[user1[x]] - smoking_mapping[user2[x]]
        elif user1[x] in alcohol_mapping and user2[x] in alcohol_mapping:
            temp += alcohol_mapping[user1[x]] - alcohol_mapping[user2[x]]
        elif user1[x] in punctuality_mapping and user2[x] in punctuality_mapping:
            temp += punctuality_mapping[user1[x]] - punctuality_mapping[user2[x]]
        elif user1[x] in lying_mapping and user2[x] in lying_mapping:
            temp += lying_mapping[user1[x]] - lying_mapping[user2[x]]
        elif user1[x] in internet_mapping and user2[x] in internet_mapping:
            temp += internet_mapping[user1[x]] - internet_mapping[user2[x]]
        elif user1[x] in gender_mapping and user2[x] in gender_mapping:
            temp += gender_mapping[user1[x]] - gender_mapping[user2[x]]
        elif user1[x] in handed_mapping and user2[x] in handed_mapping:
            temp += handed_mapping[user1[x]] - handed_mapping[user2[x]]
        elif user1[x] in education_mapping and user2[x] in education_mapping:
            temp += education_mapping[user1[x]] - education_mapping[user2[x]]
        elif user1[x] in onlychild_mapping and user2[x] in onlychild_mapping:
            temp += onlychild_mapping[user1[x]] - onlychild_mapping[user2[x]]
        elif user1[x] in city_mapping and user2[x] in city_mapping:
            temp += city_mapping[user1[x]] - city_mapping[user2[x]]
        elif user1[x] in house_mapping and user2[x] in house_mapping:
            temp += house_mapping[user1[x]] - house_mapping[user2[x]]
    
        temp = temp**2
        distance += temp

    distance = math.sqrt(distance)

    return distance

# print(distance(dataset.iloc[0], dataset.iloc[1]))
