#Machine Learning - Uniform Estimator code


#assigning global variables
n <- 100
L <- 10

#create method to generate random samples for uniform distribuion and compute L-mom & L-mle
Estimates <- function()
{
  X <- runif(n, min = 0, max = L)
  L_mom = (2 * sum(X))/n
  L_mle = max(X)
  cat('MOM Estimate ',L_mom,' & MLE Estimate',L_mle)
}

for(j in 1:1000)
{
  cat('\n\nX[',j,'] - ')
  Estimates()
}


#compute the MSE for Estimates of MOM & MLE
X <- runif(n, min = 0, max = L)
L_mom = (2 * sum(X))/n
L_mle = max(X)
cat('MOM Estimate ',L_mom,' MLE Estimate',L_mle)

MSE_mom = (L_mom - L) ^ 2
MSE_mle = (L_mle - L) ^ 2
cat('MSE for MOM Estimate ',MSE_mom,' & MLE Estimate',MSE_mle)


