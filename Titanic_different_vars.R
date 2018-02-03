#Titanic different variables
##Loading data
install.packages("dummies")

require(data.table)
require(stringr)
require(dummies)
require(randomForest)
require(rpart)
library(xgboost)

main_titanic <- fread("F:\\Kaggle stuff\\titanic\\train.csv")


wrangle_data <- function(main_titanic, type){
  
  #Imputing NAs in age with average age of mens and women
  main_titanic_male_age <- mean(main_titanic[main_titanic$Sex == 'male']$Age, na.rm = TRUE)
  main_titanic_female_age <- mean(main_titanic[main_titanic$Sex == 'female']$Age, na.rm = TRUE)
  
  main_titanic[main_titanic$Sex == 'male']$Age[is.na(main_titanic[main_titanic$Sex == 'male']$Age)] <- main_titanic_male_age
  main_titanic[main_titanic$Sex == 'female']$Age[is.na(main_titanic[main_titanic$Sex == 'female']$Age)] <- main_titanic_female_age
  
  #Getting title from name
  
  #Getting all the titles ['Mr, 'Ms', 'Dr']
  titles <- strsplit(main_titanic$Name, split = ", ")
  titles_list <- lapply(titles, function (x) x[2])
  titles_list <- unlist(titles_list)
  titles_list <- strsplit(titles_list, split = " ")
  titles_list <- lapply(titles_list, function (x) x[1])
  titles_list <- unlist(titles_list)
  
  
  #Adding a column based on their title
  for(i in 1:nrow(main_titanic)){
    if(titles_list[i] %in% c("Mr.", "Master", "Don.", "Rev.", "Major.", "Sir.", "Col.", "Capt.", "Jonkheer")){
      titles_list[i] <- "Mr"
    }else if(titles_list[i] %in% c("Mrs.", "Mme.", "Countess.")){
      titles_list[i] <- "Mrs"
    }else if(titles_list[i] %in% c("Ms.", "Mlle.")){
      titles_list[i] <- "Ms"
    }else if(titles_list[i] == 'male'){
      titles_list[i] <- "Mr"
    }else{
      titles_list[i] <- "Mrs"
    }
  }
  
  main_titanic$title <- titles_list[i]
  
  #Getting the Deck from cabin
  main_titanic[main_titanic$Cabin == '']$Cabin <- 'U'
  main_titanic <- transform(main_titanic, deck = substr(Cabin, 1, 1))
  
  #Getting family size
  main_titanic <- transform(main_titanic, family_size = SibSp + Parch + 1)
  
  #Getting fare per person
  main_titanic <- transform(main_titanic, unit_fare = Fare/family_size)
  
  #Interaction term for age and class
  main_titanic <- transform(main_titanic, age_class = Age*Pclass)
  main_titanic <- transform(main_titanic, unit_fare = Fare/family_size)
  
  #Handling Embarked 
  main_titanic[main_titanic$Embarked == '']$Embarked <- "S"
  
  #Creating dummies
  titanic <- dummy.data.frame(main_titanic, names = c("Pclass", "Sex", "Embarked", "title", "deck"), sep = ".")
  
  #Deleting variables not in use
  if(type == "train"){
    titanic <- titanic[, c(2,3,4,5,7,8,9, 10, 11, 13, c(15:27))]  
    titanic <- titanic[, -c(which(names(titanic) %in% c("deck.T")))]
  }else{
    titanic <- titanic[, c(1,2,3,4,6,7,8, 9, 10, 12, c(14:25))]  
  }
  
  return(titanic)
}

#Feature Engineering
titanic <- wrangle_data(main_titanic, "train")

#Getting Response variables out
yTRAIN <- titanic$Survived
BasetableTRAIN <- titanic[, -c(1)]

#BasetableTRAIN <- BasetableTRAIN[, -c(17)]

##Treating and Predicing the test class:
titanic_test <- fread("F:\\Kaggle stuff\\titanic\\test.csv")
BasetableTEST <- wrangle_data(titanic_test, "test")

PassengerId <- BasetableTEST$PassengerId
BasetableTEST <- BasetableTEST[, -c(1)]

#Getting some weird row with all NAs not sure why.. so have to treat that one
BasetableTEST[is.na(BasetableTEST)] <- 0



#Trying Random Forest - 76.55
rFmodel <- randomForest(x=BasetableTRAIN,
                        y=as.factor(yTRAIN),
                        ntree=1000,
                        importance=TRUE)

varImpPlot(rFmodel)
##Predicting
predrF <- predict(rFmodel,BasetableTEST)



##Trying Logistic Regression - 75
LR <- glm(yTRAIN ~ .,
          data=BasetableTRAIN,
          family=binomial("logit"))

LRstep <- step(LR, direction="both", trace = FALSE)

predLRstep <- predict(LRstep,
                      newdata=BasetableTEST,
                      type = "response")


##Trying decision tree # 80.382
ensemblesize <- 300
ensembleoftrees <- vector(mode='list',length=ensemblesize)


for (i in 1:ensemblesize){
  bootstrapsampleindicators <- sample.int(n=nrow(BasetableTRAIN),
                                          size=nrow(BasetableTRAIN),
                                          replace=TRUE)
  ensembleoftrees[[i]] <- rpart(yTRAIN[bootstrapsampleindicators] ~ .,
                                BasetableTRAIN[bootstrapsampleindicators,])
}


baggedpredictions <- data.frame(matrix(NA,ncol=ensemblesize,
                                       nrow=nrow(BasetableTEST)))

for (i in 1:ensemblesize){
  baggedpredictions[,i] <- as.numeric(predict(ensembleoftrees[[i]],
                                              BasetableTEST))
}

finalprediction <- rowMeans(baggedpredictions)


#Trying XGBoost
xgb <- xgboost(data = data.matrix(BasetableTRAIN), 
               label = yTRAIN, 
               eta = 0.1,
               max_depth = 15, 
               nround=50, 
               gamma = 0.3,
               subsample = 0.7,
               lambda = 0.3,
               colsample_bytree = 0.7,
               #seed = 1,
               eval_metric = "merror",
               objective = "multi:softmax",
               num_class = 12,
               nthread = 3
)

y_pred <- predict(xgb, data.matrix(BasetableTEST))

round(sum(y_pred))/length(y_pred)

#Writing final data frame
final_df <- data.frame(PassengerId, Survived = as.numeric(as.character(round(y_pred))))
fwrite(final_df, "F:\\Kaggle stuff\\titanic\\final_submission.csv")
