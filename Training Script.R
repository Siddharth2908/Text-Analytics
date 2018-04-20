# Set working directory to ensure the sessions are saved
setwd("C:/Siddhartha/Personal Documents/BOK/Data Science/Tuturial")
#Load source file into the session
Sourcedata<-read.csv("Spam.csv")
View(Sourcedata)
#Remove unwanted columns
Sourcedata<-Sourcedata[,1:2]
#Rename the columns 
names(Sourcedata)<-c("Category","Messages")
View(Sourcedata)
#Check data to see if there are any missing values
length(which(!complete.cases(Sourcedata)))
#Convert the data class as a "factor" which is typically the output that we would want to predict
Sourcedata$Category<-as.factor(Sourcedata$Category)
#Convert the data class as a "Character Vector" which is typically used to analyse
Sourcedata$Messages<-as.character(Sourcedata$Messages)
#Add a column which will have the length of the text
Sourcedata$TextLength<-nchar(Sourcedata$Messages)
#Explore the data to understand the spread
# 1. Number of possible outcome of a prediction inline with the categories 2.Feel of the distribution of the text data 3. Visualise the spread through loading ggplot2
prop.table(table(Sourcedata$Category))
summary(Sourcedata$TextLength)
#Before every text analytics, split the data into 1. Training Set,2.Validation Set and 3. Test set or a minimum of 2 - Training and Test.
# To do this, load Caret package and then split the data. The training data will be used for creating a model and test will be to check whether the model works.
#This will be a 70%-30% split
#Set the random seed for data reproducability during every re-run. This is important as otherwise, the 70%-30% split will vary
library(caret)
set.seed(32984)
indexes<-createDataPartition(Sourcedata$Category, times=1, p=0.7, list = FALSE)
Train<-Sourcedata[indexes,]
Test<-Sourcedata[-indexes,]
prop.table(table(Train$Category))
prop.table(table(Test$Category))
#Create data pipelines to create and train the model. Perform Data exploration, Pre-processing and wrangling
#Load quanteda 
#Tokenize the text, remove stopwords, symbols, punctuations, hyphens, numbers. These are based on the context of the text under analysis
#Change everyt term/token into a lower case and stem them to remove same words with various tenses
library("quanteda")
#Tokenize the text using bi-grams and Create TF.IDF matrix using bi-grams
Train.token<-tokens(Train$Messages, what = "word",remove_numbers = TRUE,remove_punct = TRUE, remove_symbols = TRUE, remove_hyphens = TRUE)
Train.token<-tokens_tolower(Train.token)
Train.token<-tokens_select(Train.token, stopwords(), selection = "remove")
Train.token<-tokens_wordstem(Train.token, language = "english")
#Train.token<-tokens_ngrams(Train.token, n=1:2)
#Convert the data into a class for Document Frequency Matrix (DFM)
Train.DFM<-dfm(Train.token, tolower = FALSE, remove = stopwords())
#Create DFM
Train.Matrix<-as.matrix(Train.DFM)
View(Train.Matrix)
TF<-function(row){
  row/sum(row)}
IDF<-function(col){
  Totalsize<-length(col)
  doccount<-length(which(col>0))
  log10(Totalsize/doccount)}
TF.IDF<-function(x,idf){
  x*idf
}
#Normalize all documents via TF
Train.token.df<-apply(Train.Matrix,1,TF)
Train.token.idf<-apply(Train.Matrix,2,IDF)
Train.token.TFIDF<-apply(Train.token.df,2,TF.IDF,idf=Train.token.idf)
#Invoke IRLBA for creating Singular Value Decomposition matrix using Latent Semantic Analysis. This is done for first 300 lexical
#words identified through a maximum of 600 iterations. Note: 600 is industry standards for better accuracy
#library(irlba)
#Train.SVD<-irlba(t(Train.token.TFIDF), nv = 300, maxit = 600)
#With TFIDF, project the data into the semantic space ie SVD vector space.
#sigma.inverse <- 1/Train.SVD$d
#u.transpose<-t(Train.SVD$u)
#document<-Train.token.TFIDF[1,]
#document.hat<-sigma.inverse*u.transpose%*%document
#Create new feature data frame using the document semantic space created
#Train.SVD <-data.frame(Category = Train$Category, Train.SVD$v)
# Use Cross-Validation(CV) as the basic model - rPart - Single Decision Tree
#Start of Single Decision Tree rpart modelling
#Add new column to the created matrix for "Category"
Train.Matrix<-cbind(Category = Train$Category, data.frame(Train.DFM))
#Clean up unwanted or unrecognized terms in DFM 
#Load caret
library(caret)
names(Train.Matrix)<-make.names(names(Train.Matrix))
# First step to create CV based model is to create stratified folds for 10 fold cross validation with 3 repetitions. This is more a best prax approach
# Set seed to ensure reproducability
set.seed(48743)
CV.folds<-createMultiFolds(Train$Category, k=10, times=5)
cv.control<-trainControl(method = "repeatedcv", number = 10, repeats = 3, index = CV.folds)
#second step is to create a model. To do this, install "doSNOW" package to run in parallel using multi cores as the CV takes long time. Create clusters to run model in parallel
library(doSNOW)
install.packages("e1071")
library(e1071)
cl<-makeCluster(3,type="SOCK")
registerDoSNOW(cl)
#Create CV model using Single Decision Tree "rpart"
Model1.rpart<-train(Category ~ ., data = Train.Matrix, method = "rpart", trControl = cv.control, tuneLength = 7)
stopCluster(cl)
#Check the results 
Model1.rpart
#End of Single Decision Tree rpart modelling
#The accuracy of the model can be improved by converting the matrix into TF-IDF based matrix. For this the first step is to create TF, IDF and TF.IDF function. 
#The using the created function, calculate, TF value, IDF value and TF*IDF values for every term in the DFM
#Create CV model using "Random Forest"
library(doSNOW)
cl<-makeCluster(3,type="SOCK")
registerDoSNOW(cl)
Model1.RF<-train(Category ~ ., data = Train.SVD, method = "rf", trControl = cv.control, tuneLength = 7)
stopCluster(cl)
#Check the results 
Model1.RF
#End of Random Forest modelling
# Analyze the results further using "Confusion Matrix" to understand the "Sensitivity and Specificity"
confusionMatrix(Train.SVD$Category, Model1.RF$finalModel$predicted)
#Add more features to improve the modelling accuracy. 
Train.SVD$TextLength <- Train$TextLength
#Create CV model using "Random Forest"
library(doSNOW)
cl<-makeCluster(3,type="SOCK")
registerDoSNOW(cl)
Model1.RF2<-train(Category ~ ., data = Train.SVD, method = "rf", trControl = cv.control, tuneLength = 7, importance = TRUE )
stopCluster(cl)
#Check the results 
Model1.RF2
#End of Random Forest modelling
# Analyze the results further using "Confusion Matrix" to understand the "Sensitivity and Specificity"
confusionMatrix(Train.SVD$Category, Model1.RF$finalModel$predicted)
#Compare various features and assess the relative importance using randomForest
library(randomForest)
varImpPlot(Model1.RF$finalModel)
varImpPlot(Model1.RF2$finalModel)
#The model can further be improved by using cosine similarity. The cosine of a document in vector space denotes how alike
#two documents are. This field can be engineered as a feature and projected into the SVD semantic space
#load lsa package for calculating cosine 
library(lsa)
Train.SVDCosine<-cosine(t(as.matrix(Train.SVD[,-c(1,ncol(Train.SVD))])))
# Train each text message and determine the mean cosine. This will help to determine the similarity between each category
# of messages. Per the hypothesis, the cosine similarity will be high for same category of messages and will be opposite between
# categories
Spam.indexes <- which(Train$Category == "Spam")
Train.SVDSpamsimi <- rep(0.0, nrow(Train.SVD))
for(i in 1:nrow(Train.SVD)) {
  Train.SVD$Spamsimi[i] <- mean(Train.SVDSpamsimi[i, Spam.indexes])
}
#Rerun the model using the new engineered feature of cosine similarity
library(doSNOW)
cl<-makeCluster(3,type="SOCK")
registerDoSNOW(cl)
Model1.RF3<-train(Category ~ ., data = Train.SVD, method = "rf", trControl = cv.control, tuneLength = 7, importance = TRUE )
stopCluster(cl)
#Check the results 
Model1.RF3
# Analyze the results further using "Confusion Matrix" to understand the "Sensitivity and Specificity"
confusionMatrix(Train.SVD$Category, Model1.RF$finalModel$predicted)
#Compare various features and assess the relative importance using randomForest
library(randomForest)
varImpPlot(Model1.RF2$finalModel)
varImpPlot(Model1.RF3$finalModel)