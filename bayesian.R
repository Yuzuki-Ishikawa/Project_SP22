# Final Project

# Preprocess data
library(rjags)
install.packages("mnormt")
library(mnormt)
library(MASS)
library("readxl")
setwd('C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project')
df0a <- read_excel("C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project/Eyes-closed EEG.xlsx",sheet="minimal_alpha1")
df0t <- read_excel("C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project/Eyes-closed EEG.xlsx",sheet="minimal_theta")
df1a <- read_excel("C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project/Eyes-closed EEG.xlsx",sheet="mild_alpha1")
df1t <- read_excel("C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project/Eyes-closed EEG.xlsx",sheet="mild_theta")
df2a <- read_excel("C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project/Eyes-closed EEG.xlsx",sheet="moderate_alpha1")
df2t <- read_excel("C:/Users/Yuzuki Ishikawa/Documents/Spring22/STAT 431/Final Project/Eyes-closed EEG.xlsx",sheet="moderate_theta")

# model input consideration
df0a_mean <- list()
df0a_var <- list()
df0t_mean <- list()
df0t_var <- list()
df1a_mean <- list()
df1a_var <- list()
df1t_mean <- list()
df1t_var <- list()
df2a_mean <- list()
df2a_var <- list()
df2t_mean <- list()
df2t_var <- list()

for (i in 3:ncol(df0a)){
  x = sum(df0a[,i])
  df0a_mean[i] = x/nrow(df0a)
  df0a_var[i] = var(df0a[,i])
  
  x = sum(df0t[,i])
  df0t_mean[i] = x/nrow(df0t)
  df0t_var[i] = var(df0t[,i])
  
  x = sum(df1a[,i])
  df1a_mean[i] = x/nrow(df1a)
  df1a_var[i] = var(df1a[,i])
  
  x = sum(df1t[,i])
  df1t_mean[i] = x/nrow(df1t)
  df1t_var[i] = var(df1t[,i])
  
  x = sum(df2a[,i])
  df2a_mean[i] = x/nrow(df2a)
  df2a_var[i] = var(df2a[,i])
  
  x = sum(df2t[,i])
  df2t_mean[i] = x/nrow(df2t)
  df2t_var[i] = var(df2t[,i])
}

alpha_mean <- cbind(df0a_mean, df1a_mean, df2a_mean)
rownames(alpha_mean) <- colnames(df0a)
theta_mean <- cbind(df0t_mean, df1t_mean, df2t_mean)
rownames(theta_mean) <- colnames(df0a)
alpha_var <- cbind(df0a_var, df1a_var, df2a_var)
rownames(alpha_var) <- colnames(df0a)
theta_var <- cbind(df0t_var, df1t_var, df2t_var)
rownames(theta_var) <- colnames(df0a)

# based on data distribution and previous studies, 
# theta band in the frontal region will be used
# 1-14 anterior-frontal/frontal regions

d0mean <- theta_mean[3:16,2]
d0mean <- as.numeric(d0mean)
d0var <- theta_var[3:16,2]
d0var <- as.numeric(d0var)

d1mean <- theta_mean[3:16,3]
d1mean <- as.numeric(d1mean)
d1var <- theta_var[3:16,3]
d1var <- as.numeric(d1var)

d2mean <- theta_mean[3:16,3]
d2mean <- as.numeric(d2mean)
d2var <- theta_var[3:16,3]
d2var <- as.numeric(d2var)




# make inputs
X0 <- cbind(as.matrix(df0t[,6:7]),as.matrix(df0t[,13:16]))
X0 <- cbind(0,X0,sqX0)
X1 <- cbind(as.matrix(df1t[,6:7]),as.matrix(df1t[,13:16]))
X1 <- cbind(1,X1,sqX1)
X2 <- cbind(as.matrix(df2t[,6:7]),as.matrix(df2t[,13:16]))
X2 <- cbind(2,X2,sqX2)



# Model input data X0: minimal, X1 = mild, X2 = moderate
# use 5-fold cross validation
X0_1 <- X0[1:6,]
X0_2 <- X0[7:12, ] 
X0_3 <- X0[13:18, ]
X0_4 <- X0[19:24, ]
X0_5 <- X0[25:30, ]

X1_1 <- X1[1:6,]
X1_2 <- X1[7:12, ] 
X1_3 <- X1[13:17, ]
X1_4 <- X1[18:22, ]
X1_5 <- X1[23:27, ]

X2_1 <- X2[1:6,]
X2_2 <- X2[7:12, ] 
X2_3 <- X2[13:18, ]
X2_4 <- X2[19:23, ]
X2_5 <- X2[24:28, ]
  
# make input matrix and label vector
Xtrain1 <- rbind(X0_1,X0_2,X0_3,X0_4,X1_1,X1_2,X1_3,X1_4,X2_1,X2_2,X2_3,X2_4)
Xtrain2 <- rbind(X0_2,X0_3,X0_4,X0_5,X1_2,X1_3,X1_4,X1_5,X2_2,X2_3,X2_4,X2_5)
Xtrain3 <- rbind(X0_3,X0_4,X0_5,X0_1,X1_3,X1_4,X1_5,X1_1,X2_3,X2_4,X2_5,X2_1)
Xtrain4 <- rbind(X0_4,X0_5,X0_1,X0_2,X1_4,X1_5,X1_1,X1_2,X2_4,X2_5,X2_1,X2_2)
Xtrain5 <- rbind(X0_5,X0_1,X0_2,X0_3,X1_5,X1_1,X1_2,X1_3,X2_5,X2_1,X2_2,X2_3)

Xtest1 <- rbind(X0_5,X1_5,X2_5)
Xtest2 <- rbind(X0_1,X1_1,X2_1)
Xtest3 <- rbind(X0_2,X1_2,X2_2)
Xtest4 <- rbind(X0_3,X1_3,X2_3)
Xtest5 <- rbind(X0_4,X1_4,X2_4)

labeling <- function(x){
  y <- matrix(nrow=nrow(x), ncol=3)
  for (i in 1:nrow(x)){
    if (x[i,1] == 0){
      y[i,] <- c(1,0,0)
  } else if (x[i,1] == 1){
      y[i,] <- c(0,1,0)
  } else {
      y[i,] <- c(0,0,1)
    }
  }
  return(y)
}

y <- labeling(Xtrain5)

Xtrain <- Xtrain5[,-1]
XtX <-t(Xtrain)%*%Xtrain
trainsize = nrow(Xtrain)
channels = ncol(Xtrain)

d <- list( y = y,
           X = Xtrain,
           B0 = rep(0,channels),
           Sig0 = XtX,
           n = trainsize )

inits <- list(list(tausq=1.0, alpha0=0.5, alpha1=-0.5, alpha2=0, beta0=rnorm(channels,sd=0.1), beta1=rnorm(channels,sd=0.5), beta2=rnorm(channels,sd=0.3)),
              list(tausq=0.5, alpha0=-0.5, alpha1=0, alpha2=0.5, beta0=rnorm(channels,sd=0.3), beta1=rnorm(channels,sd=0.1), beta2=rnorm(channels,sd=0.5)),
              list(tausq=0.1, alpha0=0, alpha1=0.5, alpha2=-0.5, beta0=rnorm(channels,sd=0.5), beta1=rnorm(channels,sd=0.3), beta2=rnorm(channels,sd=0.1))
)


# model
cat('model
{ 
  for (i in 1:n){
    y[i,] ~ dmulti(c(p0[i],p1[i],p2[i]),1)
    
    p0[i] = exp(q0[i])/softmax_sum[i]
    p1[i] = exp(q1[i])/softmax_sum[i]
    p2[i] = exp(q2[i])/softmax_sum[i]
    
    softmax_sum[i] = exp(q0[i]) + exp(q1[i]) + exp(q2[i])
    
    q0[i] = alpha0 + X[i,]%*%beta0[]
    q1[i] = alpha1 + X[i,]%*%beta1[]
    q2[i] = alpha2 + X[i,]%*%beta2[]
  }
  
  tausq ~ dgamma(1/2,1/2)
  sigma2 <- 1/tausq
  
  alpha0 ~ dnorm(0,0.0001)
  alpha1 ~ dnorm(0,0.0001)
  alpha2 ~ dnorm(0,0.0001)
  
  beta0 ~ dmnorm(B0, ginv*tausq*Sig0)
  beta1 ~ dmnorm(B0, ginv*tausq*Sig0)
  beta2 ~ dmnorm(B0, ginv*tausq*Sig0)
  
  ginv ~ dgamma(1/2,n/2)
  
}',file = {m = tempfile()})

m <- jags.model(m, d, inits, n.chains=3)

x <- coda.samples(m, c("alpha0", "alpha1", "alpha2", "beta0","beta1","beta2"), n.iter=100000)
gelman.plot(x, autoburnin=FALSE)

chain1<-as.matrix(x[1])
a<-apply(chain1,2,summary)
b<-summary(window(x,80000,100000))
c<-b[["statistics"]]
coeff <- c[,"Mean"]
coeff1 <- coeff[4:15]
coeff2<- coeff[16:27]
coeff3<- coeff[28:39]


dotest <- function(Xtest,coeff,coeff1,coeff2,coeff3){
  pre <- matrix(0,nrow = nrow(Xtest),ncol=5)
  ans <- Xtest[,1]
  Xtest <- Xtest[,-1]
  
  for (i in 1:nrow(Xtest)){
    pre[i,1] = Xtest[i,]%*%coeff1[] + coeff[1]
    pre[i,2] = Xtest[i,]%*%coeff2[] + coeff[2]
    pre[i,3] = Xtest[i,]%*%coeff3[] + coeff[3]
    
    label = which.max(pre[i,1:3])
    pre[i,4] = label
    pre[i,5] = ans[i] + 1
  } 
  return(pre)
}

predict5<-dotest(Xtest5,coeff,coeff1,coeff2,coeff3)
update(x,10000)
plot(x)


