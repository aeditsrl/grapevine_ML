#### load required packages ####
library(caret)
library(healthcareai)
library(Hmisc)
library(pdp)
library(ggplot2)

######################################################## POWDERY MILDEW #######################################################
#### load the example dataset for powdery mildew ####
data<- read.csv("powdery_example.csv", sep=';')
data$inf<-as.factor(data$inf)

#### dataset partitioning ####
## training dataset, test1 and test2 ##
test2<-subset(data, data$year>2017)

dat<-subset(data, data$year<=2017)
dat<- droplevels(dat)
dat$group<- paste(dat$farm_ID, dat$year, sep="_")
dat$group<-as.factor(dat$group)

d<-split_train_test(dat, inf, percent_train = 0.8, seed=1, grouping_col=group)
trainSet<-d$train

test1<-d$test

##### check correlation among variables #####
res <- rcorr(as.matrix(trainSet[,c("count_tr_14","gdd_7","perc_inf","count_tr_7_14","count_tr_0_7","cum_rain_7",
                                   "avg_max_14_7","avg_min_14_7","avg_14_7","cum_rain_14_7","psum_1","psum_2",
                                   "psum_3","psum_11","psum_12","w_min_avg","w_mean_avg","w_rain_cum","dis_sea",
                                   "dem100","doy","tavg_1","tavg_2","tavg_3","tavg_11","tavg_12")]), type="spearman")
P<-as.data.frame(res$P)
r<-as.data.frame(res$r)

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

corr<-flattenCorrMatrix(res$r, res$P)

## subset of the correlation matrix for variables with a Spearman’s correlation coefficient >0.9 (absolute value)
subset(corr, abs(corr$cor)>0.9)

## importance of variables with a filter approach calculating the area under the ROC curve
imp<-trainSet[,c("count_tr_14","gdd_7","perc_inf","count_tr_7_14","count_tr_0_7","cum_rain_7",
                 "avg_max_14_7","avg_min_14_7","avg_14_7","cum_rain_14_7","psum_1","psum_2",
                 "psum_3","psum_11","psum_12","w_min_avg","w_mean_avg","w_rain_cum","dis_sea",
                 "dem100","doy","tavg_1","tavg_2","tavg_3","tavg_11","tavg_12","inf")]
roc_imp <- filterVarImp(x = imp[,c("count_tr_14","gdd_7","perc_inf","count_tr_7_14","count_tr_0_7","cum_rain_7",
                 "avg_max_14_7","avg_min_14_7","avg_14_7","cum_rain_14_7","psum_1","psum_2",
                 "psum_3","psum_11","psum_12","w_min_avg","w_mean_avg","w_rain_cum","dis_sea",
                 "dem100","doy","tavg_1","tavg_2","tavg_3","tavg_11","tavg_12")], y = imp$inf)


# Variables removed:
# avg_max_14_7
# avg_14_7
# w_min_avg
# doy
# tavg_12

#### creating folds for a 10-fold cross-validation grouped by farm ID x year ####
set.seed(7)
folds <- groupKFold(trainSet$group, k = 10)


#### Training the algorithms (RF and C5.0) ####
metric <- "ROC"
control <- trainControl(method="cv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary,
                        savePredictions = T, index = folds)

## C5.0
set.seed(7)
fit.C5 <- train(inf~count_tr_14+gdd_7+perc_inf+count_tr_7_14+count_tr_0_7+cum_rain_7+avg_min_14_7+
                  cum_rain_14_7+psum_1+psum_2+psum_3+psum_11+psum_12+w_mean_avg+w_rain_cum+dis_sea+
                  dem100+tavg_1+tavg_2+tavg_3+tavg_11
                ,
                data=trainSet, method="C5.0", metric=metric, trControl=control)

## Random Forest
set.seed(7)
fit.rf <- train(inf~count_tr_14+gdd_7+perc_inf+count_tr_7_14+count_tr_0_7+cum_rain_7+avg_min_14_7+
                    cum_rain_14_7+psum_1+psum_2+psum_3+psum_11+psum_12+w_mean_avg+w_rain_cum+dis_sea+
                    dem100+tavg_1+tavg_2+tavg_3+tavg_11
                  ,
                  data=trainSet, method="rf", metric=metric, trControl=control)


#### evaluate model performance on training set ####
library(MLeval)
res_evalm<- evalm(list(fit.C5, fit.rf),
                  gnames=c('C5.0','RF'), positive = "inf")


#### results of C5.0 on test sets ####
pred_C5<- predict(fit.C5, test1)
pred_C5_1819<- predict(fit.C5, test2)

res_C5<-confusionMatrix(pred_C5, test1$inf, mode = "everything")
res_C5_1819<-confusionMatrix(pred_C5_1819, test2$inf, mode = "everything")

## results on 2018
pred_18<-subset(test2, test2$year==2018)
pred_18<-droplevels(pred_18)
p_test<- predict(fit.C5, pred_18)
res_1<-confusionMatrix(p_test, pred_18$inf, mode = "everything")

## results on 2019
pred_19<-subset(test2, test2$year==2019)
pred_19<-droplevels(pred_19)
p_test<- predict(fit.C5, pred_19)
res<-confusionMatrix(p_test, pred_19$inf, mode = "everything")

#### results of Random Forest on test sets  ####
pred_rf<- predict(fit.rf, test1)
pred_rf_1819<- predict(fit.rf, test2)
res_rf<-confusionMatrix(pred_rf, test1$inf, mode = "everything")
res_rf_1819<-confusionMatrix(pred_rf_1819, test2$inf, mode = "everything")

## results on 2018
pred_18<-subset(test2, test2$year==2018)
pred_18<-droplevels(pred_18)
p_test<- predict(fit.rf, pred_18)
res_1<-confusionMatrix(p_test, pred_18$inf, mode = "everything")

## results on 2019
pred_19<-subset(test2, test2$year==2019)
pred_19<-droplevels(pred_19)
p_test<- predict(fit.rf, pred_19)
res<-confusionMatrix(p_test, pred_19$inf, mode = "everything")


#### test of methods to deal with unbalanced datasets on the best performing model (C5.0) ####
# downsampling
# upsampling
# SMOTE
# ROSE

metric <- "ROC"

dw <- trainControl(method="cv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary,
                   savePredictions = T, sampling = "down", index = folds)

up <- trainControl(method="cv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary,
                   savePredictions = T, sampling = "up", index = folds)

sm <- trainControl(method="cv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary,
                   savePredictions = T, sampling = "smote", index = folds)

rose <- trainControl(method="cv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary,
                     savePredictions = T, sampling = "rose", index = folds)

set.seed(7)
fit.C5_dw <- train(inf~count_tr_14+gdd_7+perc_inf+count_tr_7_14+count_tr_0_7+cum_rain_7+avg_min_14_7+
                     cum_rain_14_7+psum_1+psum_2+psum_3+psum_11+psum_12+w_mean_avg+w_rain_cum+dis_sea+
                     dem100+tavg_1+tavg_2+tavg_3+tavg_11
                   ,
                   data=trainSet, method="C5.0", metric=metric, trControl=dw)
set.seed(7)
fit.C5_up<- train(inf~count_tr_14+gdd_7+perc_inf+count_tr_7_14+count_tr_0_7+cum_rain_7+avg_min_14_7+
                       cum_rain_14_7+psum_1+psum_2+psum_3+psum_11+psum_12+w_mean_avg+w_rain_cum+dis_sea+
                       dem100+tavg_1+tavg_2+tavg_3+tavg_11
                     ,
                     data=trainSet, method="C5.0", metric=metric, trControl=up)
set.seed(7)
fit.C5_sm <- train(inf~count_tr_14+gdd_7+perc_inf+count_tr_7_14+count_tr_0_7+cum_rain_7+avg_min_14_7+
                       cum_rain_14_7+psum_1+psum_2+psum_3+psum_11+psum_12+w_mean_avg+w_rain_cum+dis_sea+
                       dem100+tavg_1+tavg_2+tavg_3+tavg_11
                     ,
                     data=trainSet, method="C5.0", metric=metric, trControl=sm)

set.seed(7)
fit.C5_rose <- train(inf~count_tr_14+gdd_7+perc_inf+count_tr_7_14+count_tr_0_7+cum_rain_7+avg_min_14_7+
                       cum_rain_14_7+psum_1+psum_2+psum_3+psum_11+psum_12+w_mean_avg+w_rain_cum+dis_sea+
                       dem100+tavg_1+tavg_2+tavg_3+tavg_11
                     ,
                     data=trainSet, method="C5.0", metric=metric, trControl=rose)

#### evaluate the performance of the models on training set ####
library(MLeval)
res_evalm<- evalm(list(fit.C5, fit.C5_dw, fit.C5_up, fit.C5_sm, fit.C5_rose),
                  gnames=c('C5.0','C5.0_dw', 'C5.0_sm', 'C5.0_rose'), positive = "inf")

#### results of C5 on test sets ####
## downsampling
pred_C5_d<- predict(fit.C5_dw, test1)
res_C5_d<-confusionMatrix(pred_C5_d, test1$inf, mode = "everything")

## upsampling
pred_C5_u<- predict(fit.C5_up, test1)
res_C5_u<-confusionMatrix(pred_C5_u, test1$inf, mode = "everything")

## SMOTE
pred_C5_sm<- predict(fit.C5_sm, test1)
res_C5_sm<-confusionMatrix(pred_C5_sm, test1$inf, mode = "everything")

## ROSE
pred_C5_rose<- predict(fit.C5_rose, test1)
res_C5_rose<-confusionMatrix(pred_C5_rose, test1$inf, mode = "everything")


#### extracting all probabilities and metrics for each cut-off value (used in the paper to identify the cut-off value which optimized the informedness) on C5.0 with downsampling ####
t_d<-as.data.frame(res_evalm$probs$C5.0_dw)

#### results on test sets with the cut-off value which optimized the informedness ####
C5_prob_dw<- predict(fit.C5_dw, test1, type = "prob")
n<-c("C5_prob_inf", "C5_prob_no")
names(C5_prob_dw)<- (n)
test1$C5_prob_no<- C5_prob_dw$C5_prob_no
test1$C5_prob_inf<- C5_prob_dw$C5_prob_inf

cut_no<-0.4782929
cut_inf<-0.5217071

test1$pred_cut<-as.factor(ifelse(test1$C5_prob_inf>=cut_inf, "inf", "no"))
res_C5_cut<-confusionMatrix(test1$pred_cut, test1$inf, mode = "everything", positive="inf")

C5_prob_pred_dw<- predict(fit.C5_dw, test2, type = "prob")
n<-c("C5_prob_inf", "C5_prob_no")
names(C5_prob_pred_dw)<- (n)
test2$C5_prob_no<- C5_prob_pred_dw$C5_prob_no
test2$C5_prob_inf<- C5_prob_pred_dw$C5_prob_inf

test2$pred_cut<-as.factor(ifelse(test2$C5_prob_inf>=cut_inf, "inf", "no"))
res_C5_pred_cut<-confusionMatrix(test2$pred_cut, test2$inf, mode = "everything", positive="inf")

## results on 2018
pred_18<-subset(test2, test2$year==2018)
pred_18<-droplevels(pred_18)
C5_prob_dw<- predict(fit.C5_dw, pred_18, type = "prob")
n<-c("C5_prob_inf", "C5_prob_no")
names(C5_prob_dw)<- (n)
pred_18$C5_prob_no<- C5_prob_dw$C5_prob_no
pred_18$C5_prob_inf<- C5_prob_dw$C5_prob_inf
pred_18$pred_cut<-as.factor(ifelse(pred_18$C5_prob_inf>=cut_inf, "inf", "no"))
res_C5_cut<-confusionMatrix(pred_18$pred_cut, pred_18$inf, mode = "everything", positive="inf")

## results on 2019
pred_19<-subset(test2, test2$year==2019)
pred_19<-droplevels(pred_19)
C5_prob_dw<- predict(fit.C5_dw, pred_19, type = "prob")
n<-c("C5_prob_inf", "C5_prob_no")
names(C5_prob_dw)<- (n)
pred_19$C5_prob_no<- C5_prob_dw$C5_prob_no
pred_19$C5_prob_inf<- C5_prob_dw$C5_prob_inf
pred_19$pred_cut<-as.factor(ifelse(pred_19$C5_prob_inf>=cut_inf, "inf", "no"))
res_C5_cut<-confusionMatrix(pred_19$pred_cut, pred_19$inf, mode = "everything", positive="inf")

#### check the importance of variables on the best performing models #####
imp<-varImp(fit.C5_dw, scale = FALSE, metric="splits")

#### Partial dependence plots ####
p <- fit.C5_dw %>%
  partial(train = trainSet, pred.var=c("perc_inf"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P <-autoplot(p, ylab=" ") +
  theme_light() +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p1 <- fit.C5_dw %>%
  partial(train = trainSet, pred.var=c("cum_rain_7"), type=c("classification"), prob= TRUE,
            which.class= "inf", plot.engine = "ggplot2")
P1 <-autoplot(p1, ylab=" ") +
  theme_light() +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p2 <- fit.C5_dw %>%
  partial(train = trainSet, pred.var=c("gdd_7"), type=c("classification"), prob= TRUE,
              which.class= "inf", plot.engine = "ggplot2")
P2 <-autoplot(p2, ylab=" ") +
  theme_light() +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))
