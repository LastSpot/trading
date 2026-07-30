# R50d vs R50d_asym robustness

Benches: SPY/VOO, 50/50 VOO–QQQ (monthly), QQQ. No parameters retuned after looking.

## Holdout (locked)

### From 2018

- R50d: 21.36% CAGR, -20.92% max DD
- R50d_asym: 21.73% CAGR, -20.70% max DD
- SPY/VOO: 14.60% CAGR, -33.99% max DD
- 50/50 VOO–QQQ: 17.43% CAGR, -31.00% max DD
- QQQ: 20.09% CAGR, -35.12% max DD

### Last 24 months

- R50d: 19.77% CAGR, -15.70% max DD
- R50d_asym: 20.03% CAGR, -13.03% max DD
- QQQ: 22.48% CAGR, -22.77% max DD

## Neighborhood (from_2015)

- live_continuous: 21.39% CAGR, -22.92% max DD, calmar 0.93
- enter_1_exit_0: 21.39% CAGR, -22.92% max DD, calmar 0.93
- enter_2_exit_0: 22.07% CAGR, -23.14% max DD, calmar 0.95
- enter_2_exit_1: 21.97% CAGR, -22.09% max DD, calmar 0.99
- enter_3_exit_1: 21.08% CAGR, -21.57% max DD, calmar 0.98
- enter_3_exit_2: 18.96% CAGR, -24.03% max DD, calmar 0.79
- enter_2_exit_2: 21.17% CAGR, -23.44% max DD, calmar 0.90

## Placebo (from_2015)

- R50d: 21.39% CAGR, -22.92% max DD
- R50d_asym: 21.97% CAGR, -22.09% max DD
- placebo_shuffle_gate: 22.35% CAGR, -27.12% max DD
- placebo_lag_21: 19.94% CAGR, -23.46% max DD
- placebo_lag_63: 22.26% CAGR, -23.93% max DD
- placebo_random_on: 20.53% CAGR, -26.41% max DD
- asym_on_shuffled: 20.88% CAGR, -28.05% max DD

## Verdict

On holdout from_2018, R50d_asym vs R50d: +0.37pp CAGR, +0.22pp max DD (positive DD delta = shallower). Both books beat SPY/VOO and 50/50 on CAGR and max DD; vs QQQ they have higher CAGR and much shallower drawdowns. Neighborhood fragility: on holdout_2018, enter_2_exit_2 beat enter_2_exit_1 (asym) on Calmar — asym is nearby-competitive, not uniquely optimal. Placebo: shuffled equity gate can match/beat from_2015 CAGR but with worse Calmar (gap +0.11); on holdout_2018 live R50d beats shuffle by +2.0pp CAGR and ~6pp shallower max DD — timing of the real gate matters out of sample, and diversifier sleeves carry a lot of the level. Last 24m: QQQ CAGR 22.5% tops both books on return, but R50d_asym keeps the shallowest max DD (-13.0%).

## Limitations

- Regime bins concatenate non-contiguous days; CAGR is illustrative.
- Neighborhood is discrete vote thresholds only (not a continuous grid search).
- Holdout from_2018 still overlaps some design intuition from prior research.
- Paper A/B remains the cleanest forward test.
