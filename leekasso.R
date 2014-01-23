
## Load libraries

library(lars)
library(fields)
library(sva)
library(glmnet)

## Load debiased lasso function
source("debiasLasso.R")

lassoAccuracy <- function(y,x,x2){
  l1 = lars(t(x),y)
  cv1 = cv.lars(t(x),y,plot.it=FALSE,K=5)
  sval = max(0.05,cv1$index[which.min(cv1$cv)])
  p1 = predict.lars(l1,t(x2),s=sval,mode="fraction")
  p2 = predict.lars(l1,t(x2),s=sval,mode="fraction",type="coefficients")
  return(list(acc=mean((p1$fit > 0.5)==y),nonZero=sum(p2$coefficients!=0)))
}


lassoAccuracyDebiased <- function(y,x,x2){
  fit1 <-cv.relaxed.lasso(t(x),y,nfold=5)
  p1 <- unlist(relaxed.lasso.predict(fit1,t(x2))) > 0.5
  return(list(acc=mean(p1==y)))
}




leekassoAccuracy <- function(y,x,x2){
  mod = cbind(rep(1,100),y)
  mod0 = cbind(rep(1,100))
  pValues = f.pvalue(x,mod,mod0)
  index = which(rank(pValues) <= 10)
  
  leekX = t(x[index,])
  leekX2 = t(x2[index,])
  colnames(leekX) = colnames(leekX2) = paste("Column",1:10)
  lm1 = lm(y ~ ., data = as.data.frame(leekX)) 
  p2 = predict.lm(lm1,as.data.frame(leekX2))
  return(mean((p2 > 0.5) == y))
}

lassoAcc = leekassoAcc = lassoAccD = nonZero = array(NA,dim=c(10,10,10))

set.seed(121202015)
for(n in 1:10){
  for(s in 1:10){
    for(k in 1:10){
      # Generated the data
      effects =  rnorm(n*5,sd=s/10)
      y = rep(c(0,1),each=50)
      
      x = matrix(rnorm(500*100),nrow=500)
      x[1:(n*5),] = x[1:(n*5),] + effects %*% t(y)
      
      
      x2 = matrix(rnorm(500*100),nrow=500)
      x2[1:(n*5),] = x2[1:(n*5),] +  effects %*% t(y)
      
      tmp = lassoAccuracy(y,x,x2)
      tmp2 = lassoAccuracyDebiased(y,x,x2)
      nonZero[k,s,n] = tmp$nonZero
      lassoAcc[k,s,n] = tmp$acc
      lassoAccD[k,s,n] = tmp2$acc
      leekassoAcc[k,s,n] = leekassoAccuracy(y,x,x2)
      cat(k)
    }
    print(paste("Parameter combination", n, s, "done"))
  }
}


save(lassoAcc,leekassoAcc,lassoAccD,nonZero,file="lassodata.rda")


### Create the plots
brks = seq(0.4,1,length=65)
cols = tim.colors(64)

png(file="accuracy-plot.png",height=480,width=3*480)
tmpLasso = apply(lassoAcc,c(2,3),mean)
tmpLeekasso = apply(leekassoAcc,c(2,3),mean)
tmpLassoD = apply(lassoAccD,c(2,3),mean)
par(mfrow=c(1,3),mar=c(5,5,5,5))
image(tmpLeekasso,breaks=brks,col=cols,ylab="# of features increasing bottom to top",xlab="Signal increasing left to right",cex.lab=1.5,main="Leekasso Test Accuracy")
image.plot(tmpLasso,breaks=brks,col=cols,ylab="# of features increasing bottom to top",xlab="Signal increasing left to right",cex.lab=1.5,main="Lasso Test Accuracy")
image.plot(tmpLassoD,breaks=brks,col=cols,ylab="# of features increasing bottom to top",xlab="Signal increasing left to right",cex.lab=1.5,main="Debiased Lasso Test Accuracy")
dev.off()