data {
  int<lower=1> N;
  int<lower=1> M;
  array[N] int<lower=1, upper=M> cell_line;
  vector[N] t;
  vector[N] y;
}
parameters {
  // BEGIN P1d
  vector[M] a;
  vector[M] logr;
  real mu_a;
  real mu_logr;
  real<lower=0> tau_a;
  real<lower=0> tau_logr;
  // END P1d
  real t0;
  real<lower=0> sigma;
}
model {
  // BEGIN P1d
  mu_a ~ normal(7, 3);
  mu_logr ~ normal(-1.5, 3);
  tau_a ~ normal(0, 1);
  tau_logr ~ normal(0, 1);
  a ~ normal(mu_a, tau_a);
  logr ~ normal(mu_logr, tau_logr);
  // END P1d
  t0 ~ normal(12, 10);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    // BEGIN P1d
    int i = cell_line[n];
    real m = a[i] - log1p_exp(-exp(logr[i]) * (t[n] - t0));
    // END P1d
    y[n] ~ normal(m, sigma);
  }
}
