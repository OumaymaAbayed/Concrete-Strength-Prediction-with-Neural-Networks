# Predicting Concrete Compressive Strength Using Neural Networks in R

This project uses a **multilayer feedforward neural network** built in R to predict the **compressive strength of concrete** based on its ingredients. It includes data preprocessing, model training, evaluation, and performance improvement.

---

## 📊 Dataset

The dataset `concrete_data.csv` contains **1,030 observations** with the following 9 variables:

- `cement`
- `blast_furnace_slag`
- `fly_ash`
- `water`
- `superplasticizer`
- `coarse_aggregate`
- `fine_aggregate`
- `age` (in days)
- `concrete_compressive_strength` (target variable)

---

## 🔍 Data Exploration

```r
concrete <- read.csv("concrete_data.csv")
str(concrete)
summary(concrete$concrete_compressive_strength)
sum(is.na(concrete))  # Check for missing values
```

## 🔄 Data Normalization
A min-max normalization function is used to scale the data between 0 and 1:
```
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
concrete_norm <- as.data.frame(lapply(concrete, normalize))
```
### 🧪 Data Partitioning
We split the data into:

- Training set: 75% (first 773 observations)
- Test set: 25% (remaining 257 observations)
```
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]
```

## 🧠 Model Training (Single Hidden Node)
We use the neuralnet package to train a simple neural network with:
- 8 input nodes
- 1 hidden node
- 1 output node
```
install.packages("neuralnet")
library(neuralnet)

concrete_model <- neuralnet(
  concrete_compressive_strength ~ cement + blast_furnace_slag + fly_ash + water +
  superplasticizer + coarse_aggregate + fine_aggregate + age,
  data = concrete_train
)
```
## 📈 Network Visualization
```
options(repr.plot.width = 12, repr.plot.height = 6)
plot(concrete_model, rep = "best", intercept = FALSE, show.weights = TRUE, information = TRUE)
```
## ✅ Model Evaluation
```
model_results <- compute(concrete_model, concrete_test[1:8])
predicted_strength <- model_results$net.result
cor(predicted_strength, concrete_test$concrete_compressive_strength)

Correlation: ~0.722

```
Indicates a fairly strong relationship between predicted and actual values.

## 🚀 Model Improvement (5 Hidden Nodes)
To improve performance, we increase the number of hidden nodes to 5:
```
concrete_model2 <- neuralnet(
  concrete_compressive_strength ~ cement + blast_furnace_slag + fly_ash + water +
  superplasticizer + coarse_aggregate + fine_aggregate + age,
  data = concrete_train,
  hidden = 5
)

plot(concrete_model2, rep = "best", intercept = FALSE, show.weights = TRUE, information = TRUE)
```
## 🔍 Improved Evaluation
```
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$concrete_compressive_strength)
SSE reduced: from 5.666 to 1.48

Correlation improved: from 0.72 to 0.74
```
## 📌 Conclusion
Even with a basic neural network and normalized data, we achieved a strong predictive model. Increasing the network complexity further improved performance. This demonstrates how neural networks can effectively model nonlinear relationships in real-world datasets.


## 📎 References
Dataset source: UCI Machine Learning Repository – Concrete Compressive Strength

R Package: neuralnet
