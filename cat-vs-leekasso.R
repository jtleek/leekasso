#
# comparison of the "leekasso" with CAT score + HC variable selection
#

# 12 Januar 2014   Korbinian Strimmer   
#
# modified from R code by Jeffrey Leek ( https://github.com/jtleek/leekasso )


# Diskussion of the "leekasso":
# http://simplystatistics.org/2014/01/08/the-top-10-predictor-takes-on-the-debiased-lasso-still-the-champ/
# http://simplystatistics.org/2014/01/04/repost-prediction-the-lasso-vs-just-using-the-top-10-predictors/
#
# CAT score software and references:
# http://strimmerlab.org/software/sda/




#### simulation ####


## Load required libraries

# on Bioconductor
library("sva")      # f.pvalue  

# on CRAN
library("fdrtool")  # higher criticism (HC)
library("sda")      # correlation-adjusted t-score (CAT)


# estimate accuracy of leekasso and competing approaches
estimateAccuracy = function(y,x,x2, method=c("leekasso", "CAT10", "CATHC"))
{
  method=match.arg(method)


  # leekasso - rank variables according to p-values from F-test 
  # and use the top 10 variables
  if (method=="leekasso")
  {
    require("sva")
    mod = cbind(rep(1,100),y)
    mod0 = cbind(rep(1,100))
    pValues = f.pvalue(t(x),mod,mod0) # sva
    index = which(rank(pValues) <= 10)
  }

  # rank variables by CAT score and use top 10 variables
  if (method=="CAT10")
  {
    require("sda")
    cat = catscore(x, y, diagonal=FALSE, verbose=FALSE)[,1] # sda
    index = order(cat^2, decreasing=TRUE)[1:10]
  }

  # rank variables by CAT score and use HC variable selection
  if (method=="CATHC")
  {
    require("sda")
    ranking = sda.ranking(x, y, diagonal=FALSE, verbose=FALSE)
    m = which.max( ranking[,"HC"] )   
    m = min(m, 99) # restrict to 99 predictors to allow OLS fit and prediction below     
    index = ranking[1:m,"idx"]
  }


  # refit and predict using OLS
  lm1 = lm( y ~ x[,index] )
  p2 = predict.lm(lm1, as.data.frame( x2[,index] ))
  acc = mean((p2 > 0.5) == y)

  return( acc )
}

# start simulation

leekassoAcc = catAcc = cathcAcc = array(NA,dim=c(10,10,10))

set.seed(121202015)
for(n in 1:10){
  for(s in 1:10){
    for(k in 1:10){
      # Generated the data
      effects =  rnorm(n*5,sd=s/10)
      y = rep(c(0,1),each=50)
      
      x = matrix(rnorm(500*100),nrow=500)
      x[1:(n*5),] = x[1:(n*5),] + effects %*% t(y)
      x = t(x) 
    
      x2 = matrix(rnorm(500*100),nrow=500)
      x2[1:(n*5),] = x2[1:(n*5),] +  effects %*% t(y)
      x2 = t(x2)

      leekassoAcc[k,s,n] = estimateAccuracy(y,x,x2, method="leekasso")
      catAcc[k,s,n]      = estimateAccuracy(y,x,x2, method="CAT10")
      cathcAcc[k,s,n]    = estimateAccuracy(y,x,x2, method="CATHC")

      cat(k)
    }
    print(paste("Parameter combination", n, s, "done"))
  }
}

save(leekassoAcc, catAcc, cathcAcc, file="simulation.rda")


#### plot results ####

load("simulation.rda")

# average accuracy (over 10 repeats)
tmpLeekasso = apply(leekassoAcc,c(2,3),mean)
tmpCat = apply(catAcc,c(2,3),mean)
tmpCathc = apply(cathcAcc,c(2,3),mean)

sum( tmpLeekasso < tmpCathc) #100


### Create the plots
brks = seq(0.4,1,length=65)

#  tim.colors(64) as in "fields" package
cols = c("#00008F", "#00009F", "#0000AF", "#0000BF", "#0000CF", 
 "#0000DF", "#0000EF", "#0000FF", "#0010FF", "#0020FF", "#0030FF", 
 "#0040FF", "#0050FF", "#0060FF", "#0070FF", "#0080FF", "#008FFF", 
 "#009FFF", "#00AFFF", "#00BFFF", "#00CFFF", "#00DFFF", "#00EFFF", 
 "#00FFFF", "#10FFEF", "#20FFDF", "#30FFCF", "#40FFBF", "#50FFAF", 
 "#60FF9F", "#70FF8F", "#80FF80", "#8FFF70", "#9FFF60", "#AFFF50", 
 "#BFFF40", "#CFFF30", "#DFFF20", "#EFFF10", "#FFFF00", "#FFEF00", 
 "#FFDF00", "#FFCF00", "#FFBF00", "#FFAF00", "#FF9F00", "#FF8F00", 
 "#FF8000", "#FF7000", "#FF6000", "#FF5000", "#FF4000", "#FF3000", 
 "#FF2000", "#FF1000", "#FF0000", "#EF0000", "#DF0000", "#CF0000", 
 "#BF0000", "#AF0000", "#9F0000", "#8F0000", "#800000")


png(file="cat-vs-leekasso.png",height=1*480,width=3*480)

par(mfrow=c(1,3),mar=c(5,5,5,5))

image(tmpLeekasso,breaks=brks,col=cols,ylab="# of features increasing bottom to top",xlab="Signal increasing left to right",cex.lab=1.5,main="Leekasso Test Accuracy")

image(tmpCat,breaks=brks,col=cols,ylab="# of features increasing bottom to top",xlab="Signal increasing left to right",cex.lab=1.5,main="CAT + Top 10 Test Accuracy")

image(tmpCathc,breaks=brks,col=cols,ylab="# of features increasing bottom to top",xlab="Signal increasing left to right",cex.lab=1.5,main="CAT + HC Test Accuracy")

par(mfrow=c(1,1),mar=c(5.1,4.1,4.1,2.1)) # reset to default

dev.off()


