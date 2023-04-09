#Problem Statement: 
#To predict which customers are more probable to churn (Y=1), based on the attributes to the customer 

#---------------Preparing the environment for Logistic Regression----------------------

list.of.packages <- c("caret", "ggplot2","MASS","car","mlogit","caTools","sqldf","Hmisc","aod","BaylorEdPsych","ResourceSelection","pROC","ROCR")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos="http://cran.rstudio.com/")

library(caret)     # LOGISTIC MODEL
library(ggplot2)   # VISUALIZATION
library(MASS)      # VIF CALCULATION
library(car)       # VIF CALCULATION
library(mlogit)    # LOGISTIC MODEL
library(sqldf)     # WOE & IV
library(Hmisc)     # WOE & IV
library(aod)       # WALD TEST
library(ResourceSelection)  #HOSMER LEMESHOW TEST
library(pROC)      # ROC CURVE
library(ROCR)      # ROC CURVE
library(caTools)   # TRAIN AND TEST SPLIT
#library(BaylorEdPsych) # R-SQUARE

#----------------Setting the Working Directory----------------------
Path<-"E:/Projects/TELCO CUSTOMER CHURN/3-Code"
setwd(Path)
getwd()


data<-read.csv("Fn-UseC_-Telco-Customer-Churn.csv",header = TRUE)
data1=data
head(data1)

#-------------------Basic Exploration of the data--------------------
str(data1)
summary(data1)
dim(data1)

#Changing into factor
cols_cat<-c("gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","TechSupport","Churn")
data1[cols_cat]<-lapply(data1[cols_cat],factor)
summary(data1)

data1$SeniorCitizen<-as.factor(data1$SeniorCitizen)
data1$Churn<-as.factor(data1$Churn)
str(data1)


#---------------------Missing Value Treatment (if any)------------------
data.frame(colSums(is.na(data1)))


#---------------Substituting missing values with mean-------------------
data1[is.na(data1$TotalCharges),19]=mean(data1$TotalCharges,na.rm=T)
data.frame(colSums(is.na(data1)))


#---------Splitting the data into training and test data set-------------
set.seed(144) 

spl = sample.split(data1$Churn, 0.7)
data.train = subset(data1, spl == TRUE)
str(data.train)
dim(data.train)


data.test = subset(data1, spl == FALSE)
str(data.test)
dim(data.test)


#-------------Logistic Regression Model Building----------------------------


model <- glm(Churn~., data=data.train, family=binomial())
summary(model)


# Remove the insignificant variables
model <- glm(Churn~ gender +	SeniorCitizen + Partner + Dependents + tenure +	PhoneService + MultipleLines
             + InternetService + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV 
             + StreamingMovies + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges 
             + TotalCharges, data=data.train, family=binomial())
summary(model)


# Remove the insignificant variables
model <- glm(Churn~ gender +	SeniorCitizen + Partner + Dependents + tenure +	PhoneService 
             + I(MultipleLines=="Yes")+ InternetService + I(OnlineSecurity=="Yes")+ I(OnlineBackup=="Yes") 
             + I(DeviceProtection=="Yes") + I(TechSupport=="Yes") + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges +	TotalCharges 
             , data=data.train, family=binomial())
summary(model)

# Remove Partner, PhoneService, gender 
model <- glm(Churn~ 	SeniorCitizen  + tenure + Dependents
             + I(MultipleLines=="Yes")+ InternetService + I(OnlineSecurity=="Yes")+ I(OnlineBackup=="Yes") 
             + I(DeviceProtection=="Yes") + I(TechSupport=="Yes") + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges +	TotalCharges 
             , data=data.train, family=binomial())
summary(model)

# Remove  I(OnlineBackup=="Yes"), I(DeviceProtection=="Yes") , I(OnlineSecurity=="Yes")
model <- glm(Churn~ 	SeniorCitizen  + tenure 
             + I(MultipleLines=="Yes")+ InternetService   +  I(TechSupport=="Yes") + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges +	TotalCharges 
             , data=data.train, family=binomial())
summary(model)

# Remove   I(TechSupport=="Yes")
model <- glm(Churn~ 	SeniorCitizen  + tenure 
             + I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + I(PaymentMethod=="Electronic check")+	tenure , data=data.train, family=binomial())
summary(model)

#------------------Final Model---------------------------------------
# Remove  I(PaymentMethod == "Credit card (automatic)")


vif(model)

# Deviance is -2*Log Likelyhood
# AIC = -2LL + 2k
# BIC = -2LL + 2k x log(n)



# Remove   MonthlyCharges, tenure
# model <- glm(Churn~ 	SeniorCitizen  
#              + I(MultipleLines=="Yes")+ InternetService  +tenure  + I(StreamingTV=="Yes") 
#              + I(StreamingMovies=="Yes") + Contract + PaperlessBilling
#              + I(PaymentMethod=="Electronic check")+MonthlyCharge, data=data.train, family=binomial())
# summary(model)

vif(model)


model <- glm(Churn~ 	SeniorCitizen  
              + I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
              + I(StreamingMovies=="Yes") + Contract + PaperlessBilling
              + I(PaymentMethod=="Electronic check")+tenure, data=data.train, family=binomial())
summary(model)


vif(model)



#------------------------------Checking the overall fitness of the model----------------------------


#---------------Using Wald's Test--------------------
wald.test(b=coef(model), Sigma= vcov(model), Terms=1:12)
#Here Terms refer to the no. of independent variables in the final train model
#Since, p-value is less then 0.001, hence we reject Ho that all Bi=0


#------Lagrange Multiplier or Score Test (Assess whether the current variables 
# significantly improves the model fit or not) ------------------------------


# Difference between null deviance and deviance
modelChi <- model$null.deviance - model$deviance
modelChi

# Finding the degree of freedom for Null model and model with variables
chidf <- model$df.null - model$df.residual
chidf


# With more decimal places
# If p value is less than .05 then we reject the null hypothesis that the model is no better than chance.
chisq.prob <- 1 - pchisq(modelChi, chidf)
format(round(chisq.prob, 2), nsmall = 5)



#----------Lackfit Deviance for assessing whether the model where
#Ho: Observed Frequencies/probabilities =Expected Frequencies/probabilities --------

# deviance residuals
residuals(model)

# pearson residuals
residuals(model, "pearson")

sum(residuals(model, type = "pearson")^2)

deviance(model)

# Larger p value indicate good model fit
1-pchisq(deviance(model), df.residual(model))
# Thus, we accept the Null Hypothesis Ho that Observed Frequencies = Expected Frequencies





# ---------------------------------------------------------------------------------------------------
# Coefficients (Odds)
model$coefficients

# Coefficients (Odds Ratio)
exp(model$coefficients)

# Interpretation :
# if the customer has Internet Service = Fiber Optic, then he is 2.53 times more likely to churn

# Variable Importance of the model
varImp(model)

# Predicted Probabilities
prediction <- predict(model,newdata = data.train,type="response")

# Classification Model:
#prediction <- predict(model,newdata = data.train,probability=TRUE)

prediction
write.csv(prediction,"pred.csv")


rocCurve   <- roc(response = data.train$Churn, predictor = factor(prediction, ordered =TRUE), 
                  levels = rev(levels(data.train$Churn)))
data.train$Churn <- as.factor(data.train$Churn)

# Metrics - Fit Statistics
threshold<-as.numeric(coords(rocCurve,"best")[1])
predclass <-ifelse(prediction>threshold,1,0)

# Confusion Matrix
Confusion <- table(Predicted = predclass,Actual = data.train$Churn)
# Accuracy
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
# Gini coefficient
Gini <-2*auc(rocCurve)-1
# Gini coefficient is the area under the Lorenz Curve 
# (Similar ROC Curve where final model compared to baseline model)
# Range 0.5 - 0.8

# Display the confusion matrix
Confusion
# Display the Area Under the Curve
auc(rocCurve)
# Plot the ROC Curve
plot(rocCurve)


#-------------- KS (Kolmogorov-Smirnov ) statistics calculation ----------------------
data.train$m1.yhat <- predict(model, data.train, type = "response")
m1.scores <- prediction(data.train$m1.yhat, data.train$Churn)

plot(performance(m1.scores, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")

m1.perf <- performance(m1.scores, "tpr", "fpr")
ks1.logit <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit 

# Thumb rule : should lie between 0.4 - 0.7

# -------------------------------------------------------------------------------------


#----------------------------Residual Analysis ----------------------------------------

logistic_data <- data.train

logistic_data$predicted.probabilities<-fitted(model)

logistic_data$standardized.residuals<-rstandard(model)
logistic_data$studentized.residuals<-rstudent(model)
logistic_data$dfbeta<-dfbeta(model)
logistic_data$dffit<-dffits(model)
logistic_data$leverage<-hatvalues(model)

logistic_data[, c("leverage", "studentized.residuals", "dfbeta")]
write.csv(logistic_data, "Res.csv")


# ----------------------   Model has been build  ------------------------------------
# ----------------------Testing on the test dataset ---------------------------------

# Logistic Regression on full data


modelt <- glm(Churn~ 	SeniorCitizen  
              + I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
              + I(StreamingMovies=="Yes") + Contract + PaperlessBilling
              + I(PaymentMethod=="Electronic check")+	TotalCharges , data=data.test, family=binomial())
summary(modelt)

modelt <- glm(Churn~ I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
              + I(StreamingMovies=="Yes") + Contract + PaperlessBilling
              + I(PaymentMethod=="Electronic check")+	TotalCharges , data=data.test, family=binomial())
summary(modelt)



vif(modelt)

# Deviance is -2*Log Likelihood
# AIC = -2LL + 2k
# BIC = -2LL + 2k x log(n)



library(car)
library(mlogit)

# Difference between -2LL of Null model and model with variables
modelChi <- modelt$null.deviance - modelt$deviance
modelChi

# Finding the degree of freedom for Null model and model with variables
chidf <- modelt$df.null - modelt$df.residual
chidf

# With more decimal places
# If p value is less than .05 then we reject the null hypothesis that the model is no better than chance.
chisq.prob <- 1 - pchisq(modelChi, chidf)
format(round(chisq.prob, 2), nsmall = 5)


# Hosmer and Lemeshow R square
R2.hl<-modelChi/modelt$null.deviance
R2.hl


# Cox and Snell R Square (the last number; here is 2000 should be total no. of observation)

R.cs <- 1 - exp ((modelt$deviance - modelt$null.deviance) /2000)
R.cs

# Max re-scaled R square (Nagelkarke) (the last number; here is 2000 should be total no. of observation)

R.n <- R.cs /(1-(exp(-(modelt$null.deviance/2000))))
R.n



#----------Lackfit Deviance for assessing whether the model where
# Ho: Observed Frequencies/probabilities =Expected Frequencies/probabilities --------

# deviance residuals
residuals(modelt)

# pearson residuals
residuals(modelt, "pearson") 

sum(residuals(modelt, type = "pearson")^2)
deviance(modelt)

# -------------Large p value indicate good model fit----------------------------------
1-pchisq(deviance(modelt), df.residual(modelt))

# -------------------------------------------------------------------------------------
# Function - HS Test

hosmerlem <- function (y, yhat, g = 10) {
  cutyhat <- cut(yhat, breaks = quantile(yhat, probs = seq(0, 1, 1/g)),
                 include.lowest = TRUE)
  obs <- xtabs(cbind(1 - y, y) ~ cutyhat)
  expect <- xtabs(cbind(1 - yhat, yhat) ~ cutyhat)
  chisq <- sum((obs - expect)^2 / expect)
  P <- 1 - pchisq(chisq, g - 2)
  c("X^2" = chisq, Df = g - 2, "P(>Chi)" = P)
}
# -------------------------------------------------------------------------------------
# How to use the function:
# 1> Data.train is the name of the data-set. 
# 2> model is the name of the glm output
# High p value indicates the model fits well

hosmerlem(y = data.test$Churn, yhat = fitted(modelt))
# -------------------------------------------------------------------------------------
# Hosmer and Lemeshow test in a different way
# High p value incidates the model fits well

library(ResourceSelection)
hl <- hoslem.test(data.test$Churn, fitted(modelt), g=10)
hl
# -------------------------------------------------------------------------------------

# Coefficients (Odds)
modelt$coefficients

# Coefficients (Odds Ratio)
exp(modelt$coefficients)

# Predicted Probabilities
prediction <- predict(modelt,newdata = data.test,type="response")
prediction



rocCurve   <- roc(response = data.test$Churn, predictor = prediction, 
                  levels = rev(levels(data.test$Churn)))
data.test$Churn <- as.factor(data.test$Churn)


# Metrics - Fit Statistics

predclass <-ifelse(prediction>coords(rocCurve,"best")[1],1,0)
# Confusion Matrix
Confusion <- table(Predicted = predclass,Actual = data.test$Churn)
# Accuracy
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
# Gini coefficient
Gini <-2*auc(rocCurve)-1

AUCmetric <- data.frame(c(coords(rocCurve,"best"),AUC=auc(rocCurve),AccuracyRate=AccuracyRate,Gini=Gini))
AUCmetric <- data.frame(rownames(AUCmetric),AUCmetric)
rownames(AUCmetric) <-NULL
names(AUCmetric) <- c("Metric","Values")
AUCmetric

Confusion 
plot(rocCurve)



#-------------- KS (Kolmogorov-Smirnov ) statistics calculation ----------------------
data.test$m1.yhat <- predict(modelt, data.test, type = "response")

library(ROCR)
m1.scores <- prediction(data.test$m1.yhat, data.test$Churn)

plot(performance(m1.scores, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")

m1.perf <- performance(m1.scores, "tpr", "fpr")
ks1.logit <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit 

# Thumb rule : should lie between 40 - 70

#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# ----------------------- Residual Analysis --------------------------------------------


logistic_data <- data.test

logistic_data$predicted.probabilities<-fitted(modelt)
logistic_data$standardized.residuals<-rstandard(modelt)
logistic_data$studentized.residuals<-rstudent(modelt)
logistic_data$dfbeta<-dfbeta(modelt)
logistic_data$dffit<-dffits(modelt)
logistic_data$leverage<-hatvalues(modelt)

#logistic_data[, c("leverage", "studentized.residuals", "dfbeta")]
# ---------------------------------------------------------------------------------------
