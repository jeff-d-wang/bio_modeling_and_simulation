data {
  int<lower=1> N;
  vector[N] t;
  vector[N] y;
}
parameters {
  real a;
  real logr;
  real t0;
  real<lower=0> sigma;
}
model {
  a ~ normal(7, 3);
  logr ~ normal(-1.5, 3);
  t0 ~ normal(12, 10);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    real m = a - log1p_exp(-exp(logr) * (t[n] - t0));
    y[n] ~ normal(m, sigma);
  }
}
