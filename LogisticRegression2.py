import csv
import random
import numpy as np
import matplotlib.pyplot as plt


# Set a random seed for reproducibility
random.seed(42)

# Open the CSV file and read all rows into a list
with open('minimized_spambase_54.csv', 'r') as file:
    reader = csv.reader(file)
    all_rows = list(reader)
    #all_rows = [[value for idx, value in enumerate(row) if idx not in [1, 10, 36, 45]] for row in reader]


# Shuffle the list of rows in a consistent way
random.shuffle(all_rows)

# Calculate the split indices based on the desired ratio
split_index = int(len(all_rows) * 0.2)

# Split the list into two parts
test_data = all_rows[split_index:]
train_data = all_rows[:split_index]

#Store data into four arrays, test/train and out/data
test_out = [float(row[-1]) for row in test_data]
train_out = [float(row[-1]) for row in train_data]
test_data = [[1] + [float(value) for value in row[:-1]] for row in test_data] 
train_data = [[1] + [float(value) for value in row[:-1]] for row in train_data]


#define the number of data elements in the training set
train_size = len(train_data)
test_size = len(test_data)
feature_num = len(train_data[0])

#Define alpha. 
#This value was chosen as it it is the largest value that does not make the cost diverge
alpha = 0.003

#Define epsilon
epsilon = 0.00005

#define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#calculete the cost function given theta
def cost(theta):
    sum = 0
    for i in range(train_size):
        if train_out[i] == 1:
            sum += np.log(sigmoid(np.dot(theta, train_data[i])))
        else: 
            sum += np.log(1-sigmoid(np.dot(theta, train_data[i])))
    sum /= -train_size       
    return sum    

#Initialize theta to all zeros
theta = np.zeros(feature_num)

#Store Test MSE 
Test_MSE = []

#Store Train MSE 
Train_MSE = []


#Iterate gradient descent
while True:

    temp_theta = theta.copy()

    for j in range(feature_num):
        sum = 0
        for i in range(train_size):
            sum += (sigmoid(np.dot(temp_theta, train_data[i]))-train_out[i])*train_data[i][j]
        theta[j] = temp_theta[j] - alpha*sum

    #Calculate, print, then append the MSE of the test data




#    MSE = 0
#    for i in range(test_size):
#        MSE += (sigmoid(np.dot(theta, test_data[i])) - test_out[i])**2
#
#   MSE /= test_size
#  print(MSE)
#    Test_MSE.append(MSE)


 
    #Calculate, print, then append the MSE of the train data
#    MSE = 0
#    for i in range(train_size):
#        MSE += (sigmoid(np.dot(theta, train_data[i])) - train_out[i])**2
#    MSE /= train_size
#    print(MSE)
#    Train_MSE.append(MSE)

    #If new cost has decreased by amount < epsilon, break
    if cost(temp_theta) - cost(theta) < epsilon:
        break

hits = 0
for i in range(train_size):
    if abs(sigmoid(np.dot(theta, train_data[i])) - train_out[i]) < 0.5:
        hits += 1

print(hits/train_size)

#Print out the final solution: theta
#print(theta)

#plt.plot(Test_MSE, color='blue', label='Test Data MSE')
#plt.plot(Train_MSE, color='red', label='Training Data MSE')

#plt.xlabel('Epoch')
#plt.ylabel('Mean Squared Error')
#plt.title('54 Feature MSE')
#plt.legend()
#plt.show()