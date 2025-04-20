data {
  int<lower=0> N;                // Number of observations
  int<lower=0> K;                // Number of predictors
  matrix[N, K] X;                // Predictor matrix
  int<lower=0,upper=1> y[N];     // Binary outcome
}

parameters {
  real alpha;                    // Intercept
  vector[K] beta;                // Coefficients
}

model {
  // Informative priors based on previous estimates
  alpha ~ normal(-15, 2.5);
  beta[1] ~ normal(0.1, 0.05); 
  beta[2] ~ normal(0.05, 0.01);  // ck
  beta[3] ~ normal(0.05, 0.01);  // h
  beta[4] ~ normal(0.01, 0.01);  // ld

  // Logistic regression likelihood
  y ~ bernoulli_logit(alpha + X * beta);
}