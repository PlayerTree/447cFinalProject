data {
  int<lower=0> N;              // number of observations
  int<lower=0> K;              // number of predictors
  matrix[N, K] X;              // predictor matrix
  int<lower=0, upper=1> y[N];  // binary outcome
}

parameters {
  real alpha;                  // intercept
  vector[K] beta;              // coefficients
}

model {
  // Priors
  alpha ~ normal(-5, 10);
  beta[1] ~ normal(0.1, 0.1);
  beta[2] ~ normal(0.01, 0.1);
  beta[3] ~ normal(0.01, 5);
  beta[4] ~ normal(0.01, 5);

  // Likelihood
  y ~ bernoulli_logit(alpha + X * beta);
}