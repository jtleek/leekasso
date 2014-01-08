#### Given training data finds the optimal relaxed lasso model using CV error

### inputs:
## X: training feature matrix
## y: training response vector
## nfold: number of cv folds
## nlam: number of lambda values to fit the model for
## lambda.min.ratio: the smallest lambda value as a fraction of the largest

### outputs:
## models: The best linear model (at the minimum cv lambda-value)
## active.variables: A vector containing the indices of variables used in the corresponding "best model"


cv.relaxed.lasso <- function(X, y, nfold, nlam=100, lambda.min.ratio = 0.01){
  lambda <- glmnet(X,y, nlam = nlam, lambda.min.ratio = lambda.min.ratio)$lambda
  n <- nrow(X)
  folds <- makefolds(nfold, n)
  errors <- rep(0,length(lambda))
  
  for(i in 1:nfold){
    X.train <- X[-folds[[i]],]
    y.train <- y[-folds[[i]]]

    X.test <- X[folds[[i]],]
    y.test <- y[folds[[i]]]

    fits <- relaxed.lasso.fit(X.train, y.train, lambda = lambda)
    preds <- relaxed.lasso.predict(fits, X.test)

    for(j in 1:length(preds)) errors[j] <- errors[j] + sum((preds[[j]] - y.test)^2)
  }
  min.ind <- which.min(errors)
  full.fit <- relaxed.lasso.fit(X,y,lambda)
  final.model <- list(models = list(full.fit$models[[min.ind]]), active.variables = list(full.fit$active.variables[[min.ind]]))
  return(final.model)
}

### Helper function to make the folds for cross validation

makefolds <- function(nfold, n){
  remaining <- 1:n
  folds <- list()
  for(i in 1:nfold) folds[[i]] <- vector()
  counter <- 0
  while(length(remaining) > 0){
    current.fold <- (counter %% nfold) + 1
    ind <- sample(1:length(remaining), 1)
    folds[[current.fold]] <- c(folds[[current.fold]], remaining[ind])
    remaining <- remaining[-ind]
    counter = counter + 1
  }
  return(folds)
}




#### Fits the relaxed lasso along a regularization path chosen by glmnet
### Inputs:
## X: predictor matrix
## y: reponse vector
## lambda: a user supplied sequence of lambdas

### Outputs:
## models: A list of linear models (one for each lambda-value)
## active.variables: A list of vectors. Each vector contains the indices of variables used in the corresponding linear model in "models"
relaxed.lasso.fit <- function(X,y, lambda)
{
  nlam = length(lambda)
  fit <- glmnet(X,y, lambda = lambda)
  active.variables <- apply(coef(fit)[-1,],2, function(x){which(x != 0)})
  models <- list()
  for(i in 1:length(active.variables)){
    if(length(active.variables[[i]]) == 0){
      models[[i]] = lm(y ~ 1)
    }
    else{
      X.active = X[,active.variables[[i]]]
      models[[i]] <- lm(y ~ X.active)
    }
  }
  return(list(models = models, active.variables = active.variables))
}

### input:
## fit: the ouput of "relaxed.lasso.fit"
## newX: a new feature matrix to predict responses for
### ouput:
## predictions: a list of vectors. each vector contains predictions for the corresponding "model" from the "fit" input
relaxed.lasso.predict <- function(fit, newX)
{
  predictions <- list()
  for(i in 1:length(fit$models)){

    if(length(fit$active.variables[[i]]) != 0){
      X.active = cbind(rep(1,nrow(newX)), newX[,fit$active.variables[[i]], drop = F])
      predictions[[i]] <-  X.active %*% fit$models[[i]]$coefficients
    }
    else{
      predictions[[i]] = rep(coefficients(fit$models[[i]])[1],nrow(newX))
    }
  }
  return(predictions)
}

