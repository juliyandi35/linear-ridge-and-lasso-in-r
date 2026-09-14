# --- 1. PERSIAPAN DATASET SINTETIS ---
set.seed(123)

# Jumlah observasi
n <- 200

# Variabel independen
x1 <- rnorm(n, mean = 50, sd = 10)
x2 <- rnorm(n, mean = 30, sd = 5)
x3 <- rnorm(n, mean = 70, sd = 20)

# Target variabel dengan noise
y <- 5 + 0.8 * x1 - 0.6 * x2 + 0.4 * x3 + rnorm(n, mean = 0, sd = 5)

# Data frame
data <- data.frame(y, x1, x2, x3)
head(data)

# Analisis deskriptif
summary(data)

# Split data (train-test)
library(caret)
set.seed(123)
train_index <- createDataPartition(data$y, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# --- 2. MODEL REGRESI LINEAR ---
model_lm <- lm(y ~ ., data = train_data)
summary(model_lm)
pred_lm <- predict(model_lm, newdata = test_data)

# Uji Asumsi Klasik
# Uji Normalitas
shapiro.test(model_lm$residuals)

# Uji Heteroskedastisitas (Breusch Pagan)
library(lmtest)
bptest(model_lm)

# Uji Multikolinieritas
library(car)
vif(model_lm)

# Uji Autokorelasi
dwtest(model_lm)

# --- 3. REGRESI RIDGE ---
library(glmnet)

# Matrik input untuk glmnet
x_train <- as.matrix(train_data[, -1])
y_train <- train_data$y
x_test <- as.matrix(test_data[, -1])
y_test <- test_data$y

# Ridge regression
model_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5) # alpha=0 untuk ridge
coef_ridge <- predict(model_ridge, type="coefficients", s=model_ridge$lambda.min)
coef_ridge
plot(model_ridge)
pred_ridge <- predict(model_ridge, s = model_ridge$lambda.min, newx = x_test)

# --- 4. REGRESI LASSO ---
model_lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5) # alpha=1 untuk lasso
coef_lasso <- predict(model_lasso, type="coefficients", s=model_lasso$lambda.min)
coef_lasso
plot(model_lasso)
pred_lasso <- predict(model_lasso, s = model_lasso$lambda.min, newx = x_test)

# --- 5. METRIK EVALUASI ---
# Fungsi MAPE
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted)/actual)) * 100)
}

# Fungsi AIC untuk glmnet (approximate dengan residuals)
aic_glmnet <- function(model, y, pred) {
  n <- length(y)
  rss <- sum((y - pred)^2)
  k <- length(coef(model)[coef(model) != 0])
  return(n * log(rss/n) + 2 * k)
}

# Hasil evaluasi
library(Metrics)

eval <- data.frame(
  Model = c("Linear", "Ridge", "Lasso"),
  R2 = c(
    summary(model_lm)$r.squared,
    cor(y_test, pred_ridge)^2,
    cor(y_test, pred_lasso)^2
  ),
  MAPE = c(
    mape(y_test, pred_lm),
    mape(y_test, pred_ridge),
    mape(y_test, pred_lasso)
  ),
  AIC = c(
    AIC(model_lm),
    aic_glmnet(model_ridge, y_test, pred_ridge),
    aic_glmnet(model_lasso, y_test, pred_lasso)
  )
)

print("Perbandingan Model:")
print(eval)

# --- 6. VISUALISASI PREDIKSI ---
library(ggplot2)

plot_df <- data.frame(
  Actual = y_test,
  Linear = pred_lm,
  Ridge = as.numeric(pred_ridge),
  Lasso = as.numeric(pred_lasso)
)

# Ubah ke format long
library(tidyr)
plot_long <- pivot_longer(plot_df, cols = -Actual, names_to = "Model", values_to = "Predicted")

# Plot
ggplot(plot_long, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~Model) +
  theme_minimal() +
  ggtitle("Prediksi vs Aktual dari Ketiga Model")

# --- LINE PLOT: Aktual vs Prediksi ---
# Susun data
plot_df_line <- data.frame(
  Index = 1:length(y_test),
  Actual = y_test,
  Linear = pred_lm,
  Ridge = as.numeric(pred_ridge),
  Lasso = as.numeric(pred_lasso)
)

# Konversi ke format long
plot_long_line <- pivot_longer(plot_df_line, cols = -Index, names_to = "Model", values_to = "Value")

# Plot
ggplot(plot_long_line, aes(x = Index, y = Value, color = Model)) +
  geom_line(size = 1) +
  theme_minimal() +
  labs(
    title = "Perbandingan Prediksi Model Regresi vs Data Aktual",
    x = "Index Data (Test Set)",
    y = "Nilai Target"
  ) +
  scale_color_manual(values = c("Actual" = "black", "Linear" = "blue", "Ridge" = "red", "Lasso" = "green")) +
  theme(legend.position = "bottom")

