import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt

# TODO:
# 1. Descriptive statistics for initial dataset
# 2. Write k-means clustering algorithm adjustable for different #s of variables
# 3. K-means clustering generally
# 4. K-means clustering without demographics

# Considerations for later
# 1. Some columns have empty value, such as ES47 in the excel
# 2. Number of people with 0 siblings does not match the only children count


dataset = pd.read_csv("Data/responses.csv")
#print(dataset["Only child"])
dataset.drop(dataset.columns[147], axis = 1, inplace = True)

print(dataset.describe())


# ignored --
# dataset_cleaned = dataset.dropna()
# print(dataset_cleaned.describe())

# --------------------------------------------------------------------------------------
# -----------PART 1: INITIAL DEMOGRAPHIC BREAKDOWN -------------------------------------
# --------------------------------------------------------------------------------------

# Age Breakdown
age_breakdown = dataset['Age'].value_counts()
# print("Age Summary Statistics: " + age_breakdown.to_string())

# Height Breakdown
height_buckets = pd.cut(dataset['Height'], bins=range(
    145, 210, 5), include_lowest=True)
height_bucket_counts = height_buckets.value_counts(sort=False)
# print("Height Bucket Counts:" + height_bucket_counts.to_string())

# Gender Breakdown
gender_breakdown = dataset['Gender'].value_counts()
# print("Gender Summary Statistics:" + gender_breakdown.to_string())

# Weight
weight_buckets = pd.cut(dataset['Weight'], bins=range(
    40, 170, 5), include_lowest=True)
weight_bucket_counts = weight_buckets.value_counts(sort=False)
# print("Weight Bucket Counts:" + weight_bucket_counts.to_string())

# Number of siblings
sibs_breakdown = dataset['Number of siblings'].value_counts()
# print("Sibling # Value Counts:" + sibs_breakdown.to_string())

# Left - right handed
hand_breakdown = dataset['Left - right handed'].value_counts()
# print("Left or Right Handed Value Counts:" + hand_breakdown.to_string())

# Education
ed_breakdown = dataset['Education'].value_counts()
# print("Education Value Counts:" + ed_breakdown.to_string())

# Only child
#onlychild_breakdown = dataset['Only child'].value_counts()
# print("Only Child Value Counts:" + onlychild_breakdown.to_string())

# Village - town
rural_breakdown = dataset['Village - town'].value_counts()
# print("Village or Town Value Counts:" + rural_breakdown.to_string())

# House - block of flats'
house_breakdown = dataset['House - block of flats'].value_counts()
# print("House or Block of Flats Value Counts:" + house_breakdown.to_string())

# Mapping of categorical columns to 1-5 number scales
smoking_mapping = {'never smoked': 1, 'tried smoking': 2,
                   'former smoker': 4, 'current smoker': 5}

alcohol_mapping = {'never': 1, 'social drinker': 3, 'drink a lot': 5}

punctuality_mapping = {'i am always on time': 1,
                       'i am often early': 3, 'i am often running late': 5}

lying_mapping = {'never': 1, 'only to avoid hurting someone': 3,
                 'sometimes': 4, 'everytime it suits me': 5}

internet_mapping = {'less than an hour a day': 1,
                    'few hours a day': 3, 'most of the day': 5}

gender_mapping = {'female': 0, 'male': 1}

handed_mapping = {'left handed': 0, 'right handed': 1}

education_mapping = {'primary school': 1, 'secondary school': 2,
                     'college/bachelor degree': 4, 'masters degree': 5}

onlychild_mapping = {'no': 0, 'yes': 1}

city_mapping = {'village': 0, 'city': 1}

house_mapping = {'block of flats': 0, 'house/bungalow': 1}

all_mapping = [smoking_mapping, alcohol_mapping, punctuality_mapping,
               lying_mapping, internet_mapping, gender_mapping, handed_mapping,
               education_mapping, onlychild_mapping, city_mapping, house_mapping]


# --------------------------------------------------------------------------------------
# -----------PART 2: K-MEANS CLUSTERING  -----------------------------------------------
# --------------------------------------------------------------------------------------

# data = dataset, k = # of centroids, iterations is maximum (won't reach if converge)
def lloyds_kmeans(data, k, iterations):
    # randomly initialize centroids

    centroids = data.sample(n=k).values
    # clusters = [[]*k]
    clusters = [[] for _ in range(k)]  # NEW CODE

    for i in range(iterations):
        print("Starting iteration " + str(i))

        print("Total distance before cluster reassignment: ",
              sumdistances(centroids, clusters))
        clusters = [[] for _ in range(k)]
        # calculate distance from each point to all centroids
        distances = [[0]*len(data) for x in range(k)]
        for centroidindex in range(k):
            # print("Centroid index: " + str(centroidindex))
            # NEW CODE, centroidindex was i before
            currentcentroid = centroids[centroidindex]
            for rowindex in range(len(data)):
                # print("Row index: " + str(rowindex))
                currentpoint = data.iloc[rowindex]
                # print("CURRENT CENTROID ", currentcentroid)
                # print("TYPE OF CURRENT CENTROID ", type(currentcentroid))
                distances[centroidindex][rowindex] = distance(
                    currentcentroid, currentpoint)

        # assign point to correct cluster
        for rowindex in range(len(data)):
            tempmin = 0
            for centroidindex in range(k):
                if distances[centroidindex][rowindex] < distances[tempmin][rowindex]:
                    tempmin = centroidindex
            clusters[tempmin].append(data.iloc[rowindex])
        print("Total distance before new centroids: ",
              sumdistances(centroids, clusters))

        # new centroids based on mean of each cluster -- CONNOR
        # TODO: CASE WHEN CLUSTER NULL
        centroidsnew = [meandata(cluster) for cluster in clusters]
        print(centroidsnew)

        # REPORT TOTAL DISTANCE OF ALL CLUSTERS
        print("Total distance: ", sumdistances(centroidsnew, clusters))

        # check if new centroids are same as old ones (convergence)
        if np.array_equal(centroids, centroidsnew):  # NEW CODE
            print("Converged after " + str(i) + " iterations")
            break
        centroids = centroidsnew

    # calculate distance from each point to final centroids
    distances = [[0]*len(data) for x in range(k)]
    for centroidindex in range(k):
        # currentcentroid = centroids[i] #Where is i coming from, this is outside of for loop
        currentcentroid = centroids[centroidindex]
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


def sumdistances(centroids, clusters):
    tempsum = 0
    for centerindex in range(len(centroids)):
        for pointindex in range(len(clusters[centerindex])):
            tempsum += distance(centroids[centerindex],
                                clusters[centerindex][pointindex])
    return tempsum

# mean of a collection of rows -- for clusters


def meandata(data):
    means_and_categorical_modes = []
    rows = len(data)
    cols = len(data[0]) if data else 0  # NEW CODE
    for col in range(cols):
        sum = 0
        frequencies = {}
        for row in range(rows):
            # The case for when the data is categorical (11 of the 150 columns)
            if isinstance(data[row][col], str):
                if data[row][col] in frequencies:
                    frequencies[data[row][col]] += 1
                else:
                    frequencies[data[row][col]] = 1

            # The case for when the data is an integer (139 of the 150 columns)
            else:
                if not math.isnan(data[row][col]):
                    sum += data[row][col]

        # The case for when the data is an integer (139 of the 150 columns)
        if sum != 0:
            # print("Sum: ", sum, "and Rows: ", rows)
            means_and_categorical_modes.append(
                round(float(sum) / float(rows), 4))
        # The case for when the data is categorical (11 of the 150 columns)
        else:
            tempmax = 0
            tempkey = ""
            for freq in frequencies:
                if frequencies[freq] > tempmax:
                    tempmax = frequencies[freq]
                    tempkey = freq
            means_and_categorical_modes.append(tempkey)

            # means_and_categorical_modes.append(
            #   max(frequencies, key=frequencies.get))

    # print("Means method: ")
    # print(means_and_categorical_modes)  # Testing print
    return means_and_categorical_modes


def distance(user1, user2):
    distance = 0
    # print("TYPE OF USER1: ", type(user1))

    for x in range(len(user1)):
        temp = 0

        user1val = user1[x]
        user2val = user2[x]
        
        #if x == 150-10:
         #   temp += user1val/20 - user2val/20
        #elif x == 150-9:
         #   temp += user1val/41 - user2val/41
        #elif x == 150-8:
         #   temp += user1val/33 - user2val/33
        #elif x == 150-7:
         #   temp += user1val/2 - user2val/2
        #if isinstance(user1val, np.float64) or isinstance(user1val, np.int64) or isinstance(user2val, np.float64) or isinstance(user2val, np.int64):
        #if False: 
        if not isinstance(user1val, str) and not isinstance(user2val, str):
            #print(user1val)
            #print(user2val)
            if not math.isnan(user1val) and not math.isnan(user2val):
                #print("An error has occurred and you're ugly for not inputting a resopnse")
                tempval = user1val - user2val
                if tempval > 5:
                    while(tempval > 5):
                        tempval = tempval/2
                temp += tempval
                #print("Hellppppp")
        elif isinstance(user1val, str) and isinstance(user2val, str):
            for map in all_mapping:
                if user1val in map and user2val in map:
                    temp += map[user1val] - map[user2val]
                    break
        temp = temp**2
        distance += temp

    return distance
"""
        if type(user1[x]) == str and type(user1[x]) != type(user2[x]):
            print("rkelwgjerg")
            print(x)
            print(type(user2[x]))
            print(user1val)
            print(user2val)
        if type(user1[x]) != type(user2[x]):
            user1val = np.float64(user1val)
            user2val = np.float64(user2val)
            
        
        if x == 150-10:
            temp += user1val/20 - user2val/20
        elif x == 150-9:
            temp += user1val/41 - user2val/41
        elif x == 150-8:
            temp += user1val/33 - user2val/33
        elif x == 150-7:
            temp += user1val/2 - user2val/2
            """
        
"""""
        
        elif user1[x] in smoking_mapping and user2[x] in smoking_mapping:
            temp += smoking_mapping[user1[x]] - smoking_mapping[user2[x]]
        elif user1[x] in alcohol_mapping and user2[x] in alcohol_mapping:
            temp += alcohol_mapping[user1[x]] - alcohol_mapping[user2[x]]
        elif user1[x] in punctuality_mapping and user2[x] in punctuality_mapping:
            temp += punctuality_mapping[user1[x]] - \
                punctuality_mapping[user2[x]]
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
        """""

        

    #distance = math.sqrt(distance)


def visualization(data):
    xvalues = list(range(len(data.iloc[0])))  
    
    for index in range(len(data)):
        #print(data.iloc[index])
            
        yvalues = []
        for point in range(len(data.iloc[index])):
            if isinstance(data.iloc[index][point], np.float64) or isinstance(data.iloc[index][point], np.int64):
                if point == len(data.iloc[0])-10:
                    yvalues.append(data.iloc[index][point]/20)
                elif point == len(data.iloc[0])-9:
                    yvalues.append(data.iloc[index][point]/41)
                elif point == len(data.iloc[0])-8:
                    yvalues.append(data.iloc[index][point]/33)
                elif point == len(data.iloc[0])-7:
                    yvalues.append(data.iloc[index][point]/2)
                else:
                    yvalues.append(data.iloc[index][point])      
            else:
                count = 0
                for map in all_mapping:
                    if data.iloc[index][point] in map:
                        yvalues.append(map[data.iloc[index][point]])
                        count = 1
                        break
                if count != 1:
                    yvalues.append(yvalues[len(yvalues)-1])
     
        #print(index)
        plt.plot(xvalues, yvalues, alpha = 0.5)
    plt.title(label = "Visualization of Responses")
    plt.show()

# print(distance(dataset.iloc[0], dataset.iloc[1]))


# TESTING ZONE!

testdata = dataset
nodemodata = dataset.iloc[:, :-10]

print("wehgklwejglkwejgklw")
print(nodemodata)

# print(testdata)

#resultclusters = lloyds_kmeans(testdata, 3, 100)
# print(resultclusters)



#print(testdata)
#visualization(testdata)

totalresults = lloyds_kmeans(dataset, 4, 100)

#print(distance(dataset.iloc[0], dataset.iloc[1]))

#nodemoresults = lloyds_kmeans(nodemodata, 4, 100)
