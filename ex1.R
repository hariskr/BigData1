#load data - wdbc

require(Matrix)

###Solve WLS problem

#simulate data

# create sparse matrix with simulated data
n = 1000
p = 500 
X = matrix(rnorm(n*p), nrow=n)
mask = matrix(rbinom(n*p,1,0.04), nrow=n, ncol=p)
X = mask*X
beta = runif(p)
y = X %*% beta + rnorm( n, mean = 0, sd = 1)
W <- diag(rep(1, n)) 

#inversion method
inversion <- function(y,X,W)
{
  return(solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% y)
}

#QR decomposition method
QRdec <- function(y,X,W)
{
  betahat <- 1
  
  #decomposition
  Wsqrt = diag(sqrt(diag(W)))
  QR = qr(diag(sqrt(diag(W)))%*%X)
  
  #solve R*betahat = t(Q)*Wsqrt
  QW = t(qr.Q(QR)) %*% W.sqrt %*% y
  R = qr.R(QR)            #components of decomposition
  for(j in ncol(X):1){
    index = c(2:ncol(X),0)[j:ncol(X)]
    betahat[j] = (QW[j] - sum(R[j,index]*betahat[index]))/R[j,j]
  }
  return(betahat)
}

#sparse matrix

###Gradient descent problem

#Predictor variables X ~ first 10 features
X <- as.matrix(wdbc[,c(3,12)])
#Add ones to X
X <- cbind(rep(1, nrow(X)),X)
X = scale(X)

#Response variable ~ classification as M (1) or B (0) - I modified the data file
y <- as.matrix(wdbc[,2])
m <- nrow(y)

#
#Gradient descent
#

#define gradient as per logistic regression
grad.get <- function(y, X, w, m) {
  gradient <- -t(X) %*% (y - m*w)	
  return(gradient)
}

#GD algorithm
grad.descent <- function(X, iter){
  #initialize parameters
  theta <- matrix(c(0, 0), nrow=1) 
  
  alpha = .01 # learning rate
  for (i in 1:iter) {
    theta <- theta - alpha  * grad.get(x, y, theta)   
  }
  return(theta)
  print(grad.descent(x,1000))
}

#
#Newton's method
#

m = 1
iter = 100
tol = .001
alpha = 1

newton = function(X, y, Binit, tol, m, iter, alpha)
{
  N = dim(X)[1]
  p = dim(X)[2]
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  
  loglik = rep(0,iter)
  distance = rep(0,iter)
  mvect = rep(m,N)
  
  for (i in 2:iter)
  {
    w = as.numeric(1 / (1 + exp(-X %*% Betas[i-1,]))) #get weights
    H = hessian(X,mvect,w) #Hessian
    G = grad.get(y, X, w, mvect) #gradient 
    
    #solve linear system
    solve(H,G)
    
    #find Beta step
    u = solve(t(chol(H))) %*% G
    v = solve(chol(H)) %*% u
    
    #augment betas matrix
    Betas[i,] = Beta[i,] + v
    
    #break if distance < tolerance
    distance [i] = dist(Betas[i,]-Betas[i-1,])
    if(distance[i] < tol)
    {
      return (Betas)
      break
    }
    
    #update loglikelihood
    loglik[i] = loglike(y,w,m)
    
  }
  return (Betas)
  print(Betas)
  
}
