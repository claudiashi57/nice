# ACIC 2019
# Simulating high dimensional data based on speed dating dataset 
# Susan Gruber, sgruber@putnamds.com
# July 5, 2019
# Data from Gelman
# http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/
# Step 1: modify the covariates to make them better suited for the Data challenge
# 			    For example, "preference scale" is 1 to 10 for some of the events ("waves"), 
# 				and 0 to 100 for others.  Re-scale so that it is consistent.
# Step 2:  Develop twelve DGPs based on the modified covariate set, W.
#  		    and generate 100 datasets for each, with sample size = 2000
# 				(Consider the 8378 original observations to be the source population)
#     Mod 1. Simple, straightforward, parametric main terms models  
# 	   Mod 2. Treatment and outcome are complex functions of measured covariates
#     Mod 3.  Poor overlap
#     Mod 4: Treatment effect heterogeneity, complex models.
#  	for each modification, define low, medium, and high dimensional versions 
#      with increasing number of covariates included in the models for the PS and the outcome
#------------------------------------------------------------------------------
#  R 3.6.0 changed the "sample" function
# This re-sets it to the old behavior, same as when the ACIC 2019 datasets
# were generated
if (getRversion() < '3.6.0') {
	RNGkind(sample.kind="Rounding")
}
##------------------------------------------------------------------
# Step 1. Create modified covariate set, W
#------------------------------------------------------------------
d <- read.csv("data/ACIC_2019/ACIC_2019_Generate_DGPs/SG_Generate_High_Dim_Binary/SpeedDatingData.csv", header = TRUE)
# Change "" as a value to 0, except for one column
#don't understand this
temp.corr <- d$int_corr
d[d < 0] <- 0
d$int_corr <- temp.corr

# change NAs to -1.
d[is.na(d)] <- -1

# Waves 6 - 9 must change the scale from 1-10 to 1-100 to match the others.
# Rescale preference rating for different attributes of a person (sincere, fun, etc.)

attr.cols <- grep("attr[1-9]", colnames(d))
sinc.cols <- grep("sinc[1-9]", colnames(d))
intel.cols <- grep("intel[1-9]", colnames(d))
fun.cols <- grep("fun[1-9]", colnames(d))
amb.cols <- grep("amb[1-9]", colnames(d))
shar.cols <- grep("shar[1-9]", colnames(d))	
rescale.cols <- c(attr.cols, sinc.cols, intel.cols, fun.cols, amb.cols, shar.cols)
d[d$wave > 5 & d$wave < 10, rescale.cols] <- d[d$wave > 5 & d$wave < 10, rescale.cols] *10

# Change binary column to be 0/1 instead of 1/2
d$condtn <- d$condtn - 1

# Remove id columns
# in columns 1, 2, 4, 11, 12, 44, 47
d <- d[,-c(1, 2, 4, 11, 12, 44, 47)]

# Remove the ",' in the tuition amount value.
d$tuition <- as.numeric(gsub(",","", d$tuition))

# Replace categorical variables with numeric
for (i in c(29:33, 37, 38, 39, 42)){
	d[,i] <- as.numeric(factor(d[,i], labels = 1:length(unique(d[,i]))))
}

# drop "match", "goal", and "dec" from the data 
W <- d[, - c(8, 40, 91)] # 8378 x 186
colnames(W) <- paste0("V", 1:ncol(W))
# These covariates are highly correlated with others - will be useful for IVs in the simulations


env1 <- 183
env2 <- 185


highlyCor.cols <- c(79, 86, 105, 106, 139, 142, 143, 144, 145, 153, 178, 181, 182,  184)
d[is.na(d)] <- -1

##------------------------------------------------------------------
# Step 2. Define the DGPs and generate the datasets
#------------------------------------------------------------------
# parametric regression
# Make low, med, high dimensional versions of the outcome and treatment for each of the 4 mods.
# Select covariates, and vary the amount of overlap in  models for treatment A and  outcomeY
# low: pick a random 15 columns - no highly correlated ones, use 13 for Y and 13 for A
#  - 2 predictive of Y only, 2 IVs
# med: pick a random 30 columns - 29 regular, one highly correlated, use 25 for Y and  25 for A
# 5 predictive of Y only , 5 IVs
# high: pick a random 100 columns - 97 regular, 3 highly correlated
# 90 predictive of Y, 20 predictive of A - 2  IVs

set.seed(10)
temp <- append(env1, highlyCor.cols)
temp1 <- append(env2, temp) #making sure the two environment variables are not in the set of the covariates 

regular.Cols <- (1:ncol(W))[-temp1]  # these correlations are <= .98 - env - env
low.Cols <- sample(regular.Cols, 15)

low.Y <- low.Cols[1:13]
low.A <- low.Cols[3:15]

medCols <- c(sample(regular.Cols, 29), sample(highlyCor.cols, 1))
med.Y <- medCols[1:25]
med.A <- medCols[6:30] 


highCols <- c(sample(regular.Cols, 97), sample (highlyCor.cols, 3))
high.Y <- highCols[1:90]
high.A <- highCols[72:92]

n <- nrow(W)
temp <- highlyCor.cols[!(highlyCor.cols %in% medCols)]
poss_env <- temp[!(temp %in% highCols)]

#------------------------------------------------------------------
# Mod 1. Create treatment and outcome that are main terms only in W.  
# Outcome under no exposure occurs in 25% of the entire dataset. E[E(Y | A = 0, W))] = .2502


file_path = "/data/ACIC_2019/SpeedDatingDat/Mod1/"

set.seed(10)
 W[is.na(W)] <- -1
alpha.mod1 <- runif(ncol(W), min = -1.4, max = 1.7) / apply(W, 2, max) # coef for Y
beta.mod1 <- rnorm(ncol(W), mean = .5*alpha.mod1, sd = abs(.5*alpha.mod1) ) # coef for A
logitDRS.mod1.low <-  -.655  +  as.matrix(W[,low.Y] )%*% alpha.mod1[low.Y]
logitDRS.mod1.med <-  -2.598 + as.matrix(W[,med.Y] )%*% alpha.mod1[med.Y]
logitDRS.mod1.high <-  -3.173 + as.matrix(W[,high.Y] )%*% alpha.mod1[high.Y]

 # 37% treated for all three versions
logitA.mod1.low <-  -1.771  +  as.matrix(W[,low.A] )%*% beta.mod1[low.A]
logitA.mod1.med <-  -0.904 +  as.matrix(W[,med.A] )%*% beta.mod1[med.A]
logitA.mod1.high <-  -2.798  +  as.matrix(W[,high.A] )%*% beta.mod1[high.A]
 	
# Generate 10 datasets, each containing 2000 observations sampled with replacement
set.seed(10)
niter <- 10
n.b <- 6000
for (i in 1:niter){
	# low
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod1.low)[b])
   Y <- rbinom(n.b, 1, plogis(A + logitDRS.mod1.low[b]))
   A_oracle <-  plogis(logitA.mod1.low)[b]
   Y_oracle <- plogis(A + logitDRS.mod1.low[b])
   ITE <- plogis(logitDRS.mod1.low + 1)[b] - plogis(logitDRS.mod1.low)[b]
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path, "speedDateMod1low", i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE),file =paste0(file_path, "speedDateMod1lowOracle", i, ".csv"), row.names = FALSE)
	
	# med, 
   b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod1.med)[b])
   Y <- rbinom(n.b, 1, plogis(A + logitDRS.mod1.med[b]))
   A_oracle <- plogis(logitA.mod1.med)[b]
   Y_oracle <- plogis(A + logitDRS.mod1.med[b])
   ITE <- plogis(logitDRS.mod1.med + 1)[b] - plogis(logitDRS.mod1.med)[b]

	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path, "speedDateMod1med",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE),file =paste0(file_path, "speedDateMod1medOracle", i, ".csv"), row.names = FALSE)
	
# high
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod1.high)[b])
   Y <- rbinom(n.b, 1, plogis(A + logitDRS.mod1.high[b]))
   A_oracle <- plogis(logitA.mod1.high)[b]
   Y_oracle <- plogis(A + logitDRS.mod1.high[b]) 
   ITE <- plogis(logitDRS.mod1.high + 1)[b] - plogis(logitDRS.mod1.high)[b]
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path, "speedDateMod1high",i, ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE),file =paste0(file_path, "speedDateMod1highOracle", i, ".csv"), row.names = FALSE)
}

# True population ATE for all three Mod 1 DGPs
 psi0.mod1 <- c(low = mean(plogis(logitDRS.mod1.low + 1) - plogis(logitDRS.mod1.low)),
 		med = mean(plogis(logitDRS.mod1.med + 1) - plogis(logitDRS.mod1.med)),
 		high = mean(plogis(logitDRS.mod1.high + 1) - plogis(logitDRS.mod1.high)))
 
 low_ite <- plogis(logitDRS.mod1.low + 1) - plogis(logitDRS.mod1.low)
 med_ite <- plogis(logitDRS.mod1.med + 1) - plogis(logitDRS.mod1.med)
 high_ite <- plogis(logitDRS.mod1.high + 1) - plogis(logitDRS.mod1.high)
 
 temp <- psi0.mod1
 write.csv(data.frame(temp), file= paste0(file_path, "ate_truth.csv"), row.names= FALSE)
 write.csv(data.frame(low_ite, med_ite, high_ite), file = paste0(file_path, "ite_truth.csv"), row.names = FALSE)
 

 
#------------------------------------------------------------------
# Modification 3: Poor overlap: Go back to mod1. Keep same outcome, 
# but change pscore model so that it is more extreme because of IVs
file_path <-  "/data/ACIC_2019/SpeedDatingDat/Mod3/"
set.seed(10)
IV <- highlyCor.cols[!(highlyCor.cols %in% unique(c(low.Y, med.Y, high.Y)))]
alpha.mod3 <-alpha.mod1 # coef for Y
  # coef for A
beta.mod3.IV <- rnorm(length(IV), sd = .3)
beta.mod3 <- beta.mod1 
logitDRS.mod3.low <-  -.655  +  as.matrix(W[,low.Y] )%*% alpha.mod3[low.Y]
logitDRS.mod3.med <-  -2.598 + as.matrix(W[,med.Y] )%*% alpha.mod3[med.Y]
logitDRS.mod3.high <-  -3.173 + as.matrix(W[,high.Y] )%*% alpha.mod3[high.Y]

logitA.mod3.low <-  .01 -  as.matrix(W[,low.A] )%*% (.95*beta.mod3[low.A]) + 3.5* as.matrix(W[,IV[1:2]] )%*% beta.mod3.IV[1:2]
logitA.mod3.med <- 1.6 -  as.matrix(W[,med.A] )%*% (.8* beta.mod3[med.A]) - .22 * as.matrix(W[,IV[5:9]] )%*% beta.mod3.IV[5:9]
logitA.mod3.high <- -2.4  +  as.matrix(W[,high.A] )%*% (.9*beta.mod3[high.A]) + as.matrix(W[,IV[2:14]] )%*% (1.25*beta.mod3.IV[2:14] / apply(W[,IV[2:14]], 2, sd))



set.seed(3)
niter <- 10
n.b <- 6000
for (i in 1:niter){
	# low
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod3.low)[b])
   Y <- rbinom(n.b, 1, plogis(A + logitDRS.mod3.low[b]))
   A_oracle <- plogis(logitA.mod3.low)[b]
   Y_oracle <- plogis(A + logitDRS.mod3.low[b])
   ITE <- plogis(1 + logitDRS.mod3.low[b]) - plogis(logitDRS.mod3.low[b])
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod3low",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod3lowOracle",i , ".csv"), row.names = FALSE)
	
	# med
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod3.med)[b])
   Y <- rbinom(n.b, 1, plogis(A + logitDRS.mod3.med[b]))
   A_oracle <- plogis(logitA.mod3.med)[b]
   Y_oracle <- plogis(A + logitDRS.mod3.med[b])
   ITE <- plogis(1 + logitDRS.mod3.med[b]) - plogis(logitDRS.mod3.med[b])
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod3med",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod3medOracle",i , ".csv"), row.names = FALSE)
	
	
# high
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod3.high)[b])
   Y <- rbinom(n.b, 1, plogis(A + logitDRS.mod3.high[b]))
   A_oracle <- plogis(logitA.mod3.high)[b]
   Y_oracle <- plogis(A + logitDRS.mod3.high[b])
   ITE <- plogis(1 + logitDRS.mod3.high[b]) - plogis(logitDRS.mod3.high[b])
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod3high",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod3highOracle",i , ".csv"), row.names = FALSE)
	
}
# Calculate true population ATE for Mod 3 DGPs
 psi0.mod3 <- c(low = mean(plogis(logitDRS.mod3.low + 1) - plogis(logitDRS.mod3.low)),
 		med = mean(plogis(logitDRS.mod3.med + 1) - plogis(logitDRS.mod3.med)),
 		high = mean(plogis(logitDRS.mod3.high + 1) - plogis(logitDRS.mod3.high)))

 
low_ite <- plogis(logitDRS.mod3.low + 1) - plogis(logitDRS.mod3.low)
med_ite <- plogis(logitDRS.mod3.med + 1) - plogis(logitDRS.mod3.med)
high_ite <- plogis(logitDRS.mod3.high + 1) - plogis(logitDRS.mod3.high)

temp <- psi0.mod3
write.csv(data.frame(temp), file= paste0(file_path, "ate_truth.csv"), row.names= FALSE)
write.csv(data.frame(low_ite, med_ite, high_ite), file = paste0(file_path, "ite_truth.csv"), row.names = FALSE)


#------------------------------------------------------------------
# Mod 2: treatment and outcome  are a complex function of measured covariates
# Create interaction and ratio and discontinuous and non-linear terms.
file_path <- "/data/ACIC_2019/SpeedDatingDat/Mod2/"
W.interaction <- cbind(interact.2a = apply(W[,low.Y[1:2]], 1, prod), 
                       interact.2b = apply(W[,low.A[1:2]], 1, prod),
                       interact.2c = apply(W[,c(low.A[1], low.Y[1])], 1, prod),
                       interact.3a = apply(W[,low.Y[3:5]], 1, prod) / 8, 
                       interact.3b = apply(W[,low.Y[4:6]], 1, prod)/25,
                       interact.3c = apply(W[,c(low.Y[3], low.A[4:5])], 1, prod)/2) / 100

W.ratio <-  cbind(ratio.1 = W[,low.A[1]] / W[,low.Y[2]], 
                  ratio.2 = W[,low.Y[5]] / W[,low.A[4]],
                  ratio.3 = W[,low.Y[3]] / W[,low.Y[6]],
                  ratio.4 = W.interaction[,1] / W[,low.A[13]] * 100) / 15

# Incorporate these new covariates into low, medium, and high dimensional  covariate sets
set.seed(10)
beta.mod2 <- beta.mod1 
beta.mod2.W.interaction <-  runif(ncol(W.interaction), min = .1 , max = .4) / apply(W.interaction, 2, max)
beta.mod2.W.ratio <-  runif(ncol(W.ratio), min = -1, max = 1) / apply(W.ratio, 2, max)

logitA.mod2.low <-  -1.9 +  as.matrix(W[,low.A[-(1:2)]] )%*% beta.mod2[low.A[-(1:2)]] + W.interaction[,5] * beta.mod2.W.interaction[5] + W.ratio[,1] * beta.mod2.W.ratio[1]

logitA.mod2.med <-  -1.28 +  as.matrix(W[,med.A[-(20:24)]] )%*% beta.mod2[med.A[-(20:24)]] + W.interaction[,3:4] %*%  beta.mod2.W.interaction[3:4]+ W.ratio[,c(1, 3)] %*% beta.mod2.W.ratio[c(1, 3)]

logitA.mod2.high <-  -2.3  +  as.matrix(W[,high.A[-(8:12)]] )%*% beta.mod2[high.A[-(8:12)]] + W.interaction %*%  beta.mod2.W.interaction + W.ratio %*% beta.mod2.W.ratio

set.seed(100)
alpha.mod2 <- rnorm(ncol(W), mean = 1.1 *beta.mod2, sd = abs(.5*beta.mod2) ) 
alpha.mod2.W.interaction <- rnorm(ncol(W.interaction), mean = -1* beta.mod2.W.interaction, sd = abs(.5*beta.mod2.W.interaction) ) 
alpha.mod2.W.ratio <- rnorm(ncol(W.ratio), mean = .5 *beta.mod2.W.ratio, sd = abs(.5*beta.mod2.W.ratio) ) 

logitDRS.mod2.low <-  -2.31 +  as.matrix(W[,low.Y] )%*% alpha.mod2[low.Y] +W.interaction[,5] * alpha.mod2.W.interaction[5] + W.ratio[,1] * alpha.mod2.W.ratio[1]
logitDRS.mod2.med <-  -.7 + as.matrix(W[,med.Y[-(20:22)]] )%*% alpha.mod2[med.Y[-(20:22)]] + W.interaction[,(2:4)] %*%  alpha.mod2.W.interaction[2:4]+ W.ratio[,c(1, 3)] %*% alpha.mod2.W.ratio[c(1, 3)]
logitDRS.mod2.high <-  -3.57 + as.matrix(W[,high.Y] )%*% alpha.mod2[high.Y] +  W.interaction %*%  alpha.mod2.W.interaction + W.ratio %*% alpha.mod2.W.ratio

set.seed(2)
niter <- 10
n.b <- 6000
for (i in 1:niter){
  # low
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod2.low)[b])
  Y <- rbinom(n.b, 1, plogis(1.25*A + logitDRS.mod2.low[b]))
  
  A_oracle <- plogis(logitA.mod2.low)[b]
  Y_oracle <- plogis(1.25*A + logitDRS.mod2.low[b])
  ITE <- plogis(logitDRS.mod2.low[b] + 1.25) - plogis(logitDRS.mod2.low[b])
  
  write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path, "speedDateMod2low",i , ".csv"), row.names = FALSE)
  write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod2lowOracle",i , ".csv"), row.names = FALSE)
  
  # med
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod2.med)[b])
  Y <- rbinom(n.b, 1, plogis(1.25*A + logitDRS.mod2.med[b]))
  
  A_oracle <- plogis(logitA.mod2.med)[b]
  Y_oracle <- plogis(1.25*A + logitDRS.mod2.med[b])
  ITE <- plogis(logitDRS.mod2.med[b] + 1.25) - plogis(logitDRS.mod2.med[b])
  
  write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod2med",i , ".csv"), row.names = FALSE)
  write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod2medOracle",i , ".csv"), row.names = FALSE)
  
  # high
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod2.high)[b])
  Y <- rbinom(n.b, 1, plogis(1.25*A + logitDRS.mod2.high[b]))
  
  A_oracle <- plogis(logitA.mod2.high)[b]
  Y_oracle <- plogis(1.25*A + logitDRS.mod2.high[b])
  ITE <- plogis(logitDRS.mod2.high[b] + 1.25) - plogis(logitDRS.mod2.high[b])
  
  write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod2high",i , ".csv"), row.names = FALSE)
  write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod2highOracle",i , ".csv"), row.names = FALSE)
}
# Calculate true population ATE
psi0.mod2 <- c(low = mean(plogis(logitDRS.mod2.low + 1.25) - plogis(logitDRS.mod2.low)),
               med = mean(plogis(logitDRS.mod2.med + 1.25) - plogis(logitDRS.mod2.med)),
               high = mean(plogis(logitDRS.mod2.high + 1.25) - plogis(logitDRS.mod2.high)))

low_ite <- plogis(logitDRS.mod2.low + 1.25) - plogis(logitDRS.mod2.low)
med_ite <- plogis(logitDRS.mod2.med + 1.25) - plogis(logitDRS.mod2.med)
high_ite <- plogis(logitDRS.mod2.high + 1.25) - plogis(logitDRS.mod2.high)

temp <- psi0.mod2
write.csv(data.frame(temp), file= paste0(file_path, "ate_truth.csv"), row.names= FALSE)
write.csv(data.frame(low_ite, med_ite, high_ite), file = paste0(file_path, "ite_truth.csv"), row.names = FALSE)


#------------------------------------------------------------------
# Modification 4 - like Mod 2, with treatment effect heterogeneity
file_path <- "/data/ACIC_2019/SpeedDatingDat/Mod4/"
set.seed(10)
beta.mod4 <- beta.mod1 
beta.mod4.W.interaction <-  runif(ncol(W.interaction), min = .1 , max = .4) / apply(W.interaction, 2, max)
beta.mod4.W.ratio <-  runif(ncol(W.ratio), min = -1, max = 1) / apply(W.ratio, 2, max)

logitA.mod4.low <-  -1.9 +  as.matrix(W[,low.A[-(1:2)]] )%*% beta.mod4[low.A[-(1:2)]] + W.interaction[,5] * beta.mod4.W.interaction[5] + W.ratio[,1] * beta.mod4.W.ratio[1]

logitA.mod4.med <-  -1.27 +  as.matrix(W[,med.A[-(20:24)]] )%*% beta.mod4[med.A[-(20:24)]] + W.interaction[,3:4] %*%  beta.mod4.W.interaction[3:4]+ W.ratio[,c(1, 3)] %*% (2 *beta.mod4.W.ratio[c(1, 3)])

logitA.mod4.high <-  -2.3  +  as.matrix(W[,high.A[-(8:12)]] )%*% beta.mod4[high.A[-(8:12)]] + W.interaction %*%  beta.mod4.W.interaction + W.ratio %*% beta.mod4.W.ratio

set.seed(100)
alpha.mod4 <- rnorm(ncol(W), mean = 1.1 *beta.mod4, sd = abs(.5*beta.mod4) ) 
alpha.mod4.W.interaction <- rnorm(ncol(W.interaction), mean = -1* beta.mod4.W.interaction, sd = abs(.5*beta.mod4.W.interaction) ) 
alpha.mod4.W.ratio <- rnorm(ncol(W.ratio), mean = .5 *beta.mod4.W.ratio, sd = abs(.5*beta.mod4.W.ratio) ) 

logitDRS.mod4.low <-  -2.31  +  as.matrix(W[,low.Y] )%*% alpha.mod4[low.Y] +W.interaction[,5] * alpha.mod4.W.interaction[5] + W.ratio[,1] * alpha.mod4.W.ratio[1]
logitDRS.mod4.med <-  -.676 + as.matrix(W[,med.Y[-(20:22)]] )%*% (1.5*alpha.mod4[med.Y[-(20:22)]]) + W.interaction[,(2:4)] %*%  alpha.mod4.W.interaction[2:4]+ W.ratio[,c(1, 3)] %*% alpha.mod4.W.ratio[c(1, 3)]
logitDRS.mod4.high <-  1 - as.matrix(W[,high.Y] )%*% alpha.mod4[high.Y] -  W.interaction %*%  alpha.mod4.W.interaction - W.ratio %*% alpha.mod4.W.ratio

set.seed(4)
niter <- 10
n.b <- 6000
for (i in 1:niter){
	# low
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod4.low)[b])
   Y <- rbinom(n.b, 1, plogis(A * (1 +  rowSums(W[b,low.Y[c(1, length(low.Y))]]))  + logitDRS.mod4.low[b]))
   
   A_oracle <- plogis(logitA.mod4.low)[b]
   Y_oracle <- plogis(A * (1 +  rowSums(W[b,low.Y[c(1, length(low.Y))]]))  + logitDRS.mod4.low[b])
   
   ITE <- plogis(logitDRS.mod4.low[b] + (1 +  rowSums(W[b,low.Y[c(1, length(low.Y))]]))) - plogis(logitDRS.mod4.low)[b]
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod4low",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod4lowOracle",i , ".csv"), row.names = FALSE)
	
	# med
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod4.med)[b])
   Y <- rbinom(n.b, 1, plogis(A * (-.5 - 1*  rowSums(W[b,med.Y[c(1, 13,18)]]) + .3 * W.ratio[b,3])+ logitDRS.mod4.med[b]))
   
   A_oracle <- plogis(logitA.mod4.med)[b]
   Y_oracle <- plogis(A * (-.5 - 1*  rowSums(W[b,med.Y[c(1, 13,18)]]) + .3 * W.ratio[b,3])+ logitDRS.mod4.med[b])
   ITE <- plogis((-.5 - 1*  rowSums(W[b,med.Y[c(1, 13,18)]]) + .3 * W.ratio[b,3])+ logitDRS.mod4.med[b]) -  plogis(logitDRS.mod4.med)[b]
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod4med",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path,"speedDateMod4medOracle",i , ".csv"), row.names = FALSE)
	
# high
    b <- sample(1:n, n.b, replace = TRUE)
   A <- rbinom(n.b, 1, plogis(logitA.mod4.high)[b])
   Y <- rbinom(n.b, 1, plogis(A * (1 -.3*  rowSums(W[b,high.Y[c(5:8)]]) + .6*W.ratio[b,3])+ logitDRS.mod4.high[b]))
   A_oracle <- plogis(logitA.mod4.high)[b]
   Y_oracle <- plogis(A * (1 -.3*  rowSums(W[b,high.Y[c(5:8)]]) + .6*W.ratio[b,3])+ logitDRS.mod4.high[b])
   ITE <- plogis((1 -.3*  rowSums(W[b,high.Y[c(5:8)]]) + .6*W.ratio[b,3])+ logitDRS.mod4.high[b]) - plogis(logitDRS.mod4.high)[b]
   
	write.csv(data.frame(Y, A, W[b,]), file = paste0(file_path,"speedDateMod4high",i , ".csv"), row.names = FALSE)
	write.csv(data.frame(A_oracle, Y_oracle, ITE), file = paste0(file_path, "speedDateMod4highOracle",i , ".csv"), row.names = FALSE)
	
}
# Calculate true population ATE for  Mod 4
 psi0.mod4 <- c(low = mean(plogis(logitDRS.mod4.low + (1 +  rowSums(W[,low.Y[c(1, length(low.Y))]]))) - plogis(logitDRS.mod4.low)),
 		med = mean(plogis(logitDRS.mod4.med +  (-.5 - 1*  rowSums(W[,med.Y[c(1, 13,18)]]) + .3*W.ratio[,3]))  - plogis(logitDRS.mod4.med)),
 		high = mean(plogis(logitDRS.mod4.high + (1 -.3*  rowSums(W[,high.Y[c(5:8)]]) + .6*W.ratio[,3])) - plogis(logitDRS.mod4.high)))
 
 
 low_ite <- plogis(logitDRS.mod4.low + (1 +  rowSums(W[,low.Y[c(1, length(low.Y))]]))) - plogis(logitDRS.mod4.low)
 med_ite <- plogis(logitDRS.mod4.med +  (-.5 - 1*  rowSums(W[,med.Y[c(1, 13,18)]]) + .3*W.ratio[,3]))  - plogis(logitDRS.mod4.med)
 high_ite <- plogis(logitDRS.mod4.high + (1 -.3*  rowSums(W[,high.Y[c(5:8)]]) + .6*W.ratio[,3])) - plogis(logitDRS.mod4.high)
 
 temp <- psi0.mod4
 write.csv(data.frame(temp), file= paste0(file_path, "ate_truth.csv"), row.names= FALSE)
 write.csv(data.frame(low_ite, med_ite, high_ite), file = paste0(file_path, "ite_truth.csv"), row.names = FALSE)


			