# Multiple Linear Regression

# Importando o dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 1:3]

#Encodando dados categóricos (ex. países, purchased (yes, no))

dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

# Dividindo o dataset em training dataset e test dataset
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting MLR to the Training set
regressor = lm(formula = Profit ~ ., # R.D.Spend + Administration + Marketing.Spend + State)
               data = training_set)  #  (O '.' indica todas as vars independentes)

# Predicting the test set results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)


library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$R.D.Spend, y = training_set$Profit), 
             color = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Profit vs R.D.Spend (Training set)') +
  xlab('R.D.Spend') +
  ylab('Profit')

# Visualizing Test set results
ggplot() + 
  geom_point(aes(x = test_set$R.D.Spend, y = test_set$Profit), 
             color = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Profit vs R.D.Spend (Test set)') +
  xlab('R.D.Spend') +
  ylab('Profit')
