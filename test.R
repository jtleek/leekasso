source("debiasLasso.R")


n <- 100
p <- 50

X <- matrix(rnorm(n*p), ncol = p)
y <- rnorm(n) + X[,1]

ind.train <- sample(1:n, n/2, replace = FALSE)

X.train <- X[ind.train,]
y.train <- y[ind.train]

X.test <- X[-ind.train,]
y.test <- y[-ind.train]

fit <- cv.relaxed.lasso(X.train, y.train, nfold = 10)

pred <- relaxed.lasso.predict(fit, X.test)

error <- sum((y.test - unlist(pred))^2)
