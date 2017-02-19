# Data Preprocessing

# Importando o dataset
dataset = read.csv('Data.csv')
dataset = dataset[, 2:3]

# Dividindo o dataset em training dataset e test dataset
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Featured Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


