library(ElemStatLearn)
library("class")
# KNN
setwd("../Dropbox/GM5/LH/Statistical Learning/")
don<-read.csv2("Capteurs.csv")

# Standardising the data
don_scale = scale(don[,c(1,2)])

# Splitting the data
train = don_scale[0:99,]
test = don_scale[100:120,]
train_X_std = train[,c(1,2)]
test_X_std = test[,c(1,2)]
train_Y = don[0:99,3]
test_Y = don[100:120,3]

knn = knn(train=train_X_std ,test= test_X_std , cl= train_Y, k = 5,prob = TRUE)
prob <- attr(knn, "prob")
prob <- ifelse(knn=="1", prob, 1-prob)
px1 =  seq(min(don$x1), max(don$x1) , 0.2)
px2 <- seq(min(don$x2), max(don$x2) , 0.2)
knnb <- matrix(prob, length(px1), length(px2))
contour(px1, px2, knnb, levels=0.02, labels="", xlab="", ylab="", main=
          "15-nearest neighbour", axes=FALSE)

