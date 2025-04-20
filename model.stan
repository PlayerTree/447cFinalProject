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
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);

  // Likelihood
  y ~ bernoulli_logit(alpha + X * beta);
}