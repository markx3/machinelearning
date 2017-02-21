# Polynomial Regression

# Importando o dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_reg)
predict(lin_reg)
predict(poly_reg)

# Visualizing SLR 
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Levels vs Salaries (Lin reg)') +
  xlab('Level') +
  ylab('Salary')

# Visualizing PR
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Levels vs Salaries (Poly Reg)') +
  xlab('Level') +
  ylab('Salary')

# Predicting with SLR
y_pred = predict(lin_reg, data.frame(Level = 6.5))
y_pred
# Predicting with PR
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))
y_pred