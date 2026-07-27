# SSTGAM architecture

SSTGAM is an additive binary classifier for wide longitudinal data. It keeps a
single effect surface for each temporal feature and evaluates that surface at
every observed time point.

## Input representation

Columns named `<feature_name>_<time_id>` become a dense tensor:

```text
values[sample, feature, time]
observed[sample, feature, time]
```

The observation mask is one for finite input values and zero for missing
values. Columns without a trailing integer are static covariates.

## Statistical model

```text
logit(p_i) = intercept
           + sum_k sum_d observed[i,k,d] * f_k(values[i,k,d], time_d)
           + sum_s g_s(static[i,s])
```

The model is additive across features and time points. It models a
time-varying effect of each individual measurement, not relationships between
measurements such as a lag, difference, or slope.

## Spline backend

The spline backend represents each temporal effect as a tensor product of
Gaussian radial basis functions:

```text
f_k(value, time) = sum_p sum_q coefficient[k,p,q]
                   * value_basis[p](value) * time_basis[q](time)
```

Second-difference penalties keep neighboring coefficients smooth along both
the value and time axes. The model is fitted by cyclic Newton boosting with
early stopping on validation log loss.

## Tree backend

The tree backend bins each feature's observed values using pooled training
quantiles. It stores a value-bin by time-bin table per feature. At each
boosting round, a shallow regression tree smooths a Newton update over that
table's value and time coordinates.

The split order is chosen per round rather than left to the tree. Depth-1
candidate splits are fitted on the value axis and the time axis separately and
scored by weighted-SSE gain minus `2 * log(m)`, where `m` is the number of
distinct candidate thresholds on that axis. The higher-scoring axis splits
first; the other axis refines within each resulting group. The penalty is a
Bonferroni/BIC-style multiplicity correction. Without it the value axis wins
on candidate count alone, since it normally has more quantile bins than there
are observation times.

Static terms have a single axis and use a plain one-stage tree.

The table is indexed by ordinal time position, but the trees split on the
normalized time coordinate, so irregular spacing between observation times is
respected when grouping time points during fitting.

## Bagging

Both backends fit `outer_bags` times on bootstrap resamples of the training
rows and average the fitted parameters. The validation split is held fixed
across bags, so early stopping is judged on the same held-out rows every time
and only the resampled training rows differ.

Averaging does not change the model form. A mean of coefficient vectors over a
shared basis is another coefficient vector, and a mean of value-time tables is
another table, so the fitted `f_k` stays exactly as defined above. Bagging is a
variance-reduction step only. Set `outer_bags=1` to disable it.

## Interpretation

Spline models expose `value_effect` and `temporal_surface`. Tree models expose
`value_time_table`. Each output is on the log-odds scale; positive values
increase the predicted probability relative to the intercept and negative
values decrease it.
