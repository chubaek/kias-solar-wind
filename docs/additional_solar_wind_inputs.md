# Additional Solar-Wind Input Candidates

Current frozen baseline:

```text
target: Speed(t + 72h)
model:  0.7 Direct MLP + 0.3 ExtraTrees
features: current tabular features + representative_mrmr_ch
```

Any new feature must be available at or before forecast origin `t`. Public
validation/CV must select features and hyperparameters. Private evaluation is
diagnostic only.

## 1. Photospheric Magnetic Field / PFSS / WSA-Like Features

### Physical Rationale

Solar-wind speed is controlled by magnetic connectivity between near-Earth
field lines and solar open-field regions. Coronal-hole area and darkness are
useful proxies, but magnetograms and PFSS/WSA-style fields provide a more direct
view of open flux, field-line expansion, source-surface footpoints, and distance
to boundaries.

### Expected Effect On 72h Speed Prediction

These features should improve high-speed stream timing and amplitude,
especially when two coronal holes have similar EUV area but different magnetic
field strength, expansion factor, or Earth-connected geometry.

### Possible Data Sources

- SDO/HMI line-of-sight or radial magnetograms, via JSOC.
- NSO/GONG synoptic or synchronic magnetograms.
- PFSS outputs generated locally from HMI/GONG maps.
- WSA-like products if available from CCMC, SWPC, or local WSA runs.

### Temporal Coverage For 2011-2025

- HMI starts in 2010 and covers the full project period.
- GONG covers the full project period and is useful as a continuity fallback.
- PFSS/WSA coverage depends on whether we generate it or download a continuous
  archive.

### Implementation Difficulty

Hard for a full PFSS/WSA pipeline. Medium if using precomputed synoptic/PFSS
products. Easy only for simple magnetogram summary features around CH masks.

### Leakage Risk

Medium to high. Synoptic maps can include far-side or future central-meridian
longitudes if constructed from a full Carrington rotation. Use only synchronic
maps available at origin `t`, or explicitly lag synoptic products so no future
longitude information enters.

### Recommended First Features

- Mean and median `|Br|` inside Earth-facing CH masks.
- Signed mean `Br` and dominant polarity inside selected CH windows.
- Total unsigned open-field proxy: `area * mean(|Br|)`.
- PFSS open-field footpoint distance to CH/open-field boundary.
- Flux-tube expansion factor along Earth-connected field line.
- HCS angular distance at source surface.

## 2. F10.7 Radio Flux

### Physical Rationale

F10.7 tracks solar EUV/UV activity and solar-cycle background. It is not a
direct high-speed stream source, but it can help model the changing background
solar wind, temperature, density, and cycle-phase distribution shift.

### Expected Effect On 72h Speed Prediction

Likely modest for short-term stream timing, but useful for bias correction and
cycle-dependent calibration, especially across Solar Cycle 24/25 changes.

### Possible Data Sources

- NOAA SWPC daily observed F10.7.
- Natural Resources Canada / Dominion Radio Astrophysical Observatory daily
  F10.7 records.
- OMNI-derived solar indices if already bundled in an OMNI product.

### Temporal Coverage For 2011-2025

Daily F10.7 has long historical coverage and should cover the full project
period.

### Implementation Difficulty

Easy.

### Leakage Risk

Low if using observed values dated at or before origin `t`. Do not use centered
running means or values from after origin.

### Recommended First Features

- Daily F10.7 at origin date.
- 27-day rolling mean.
- 81-day rolling mean.
- `F10.7(t) - rolling_mean_27d`.
- `rolling_mean_27d - rolling_mean_81d`.
- 7-day trend and 27-day trend.

## 3. CME / Flare Contamination Features

### Physical Rationale

The current model family is mainly tuned for recurrent high-speed streams.
CMEs, shocks, and flare-associated transients can cause sudden speed jumps that
coronal-hole recurrence models underpredict. The known worst case around
2024-05-12 is consistent with a transient-event failure mode.

### Expected Effect On 72h Speed Prediction

Could improve event-aware uncertainty and reduce severe underprediction during
transient disturbances. It may not improve average recurrent-stream forecasts
unless added carefully.

### Possible Data Sources

- NASA DONKI CME and flare notifications.
- CDAW LASCO CME catalog.
- GOES flare event lists and X-ray flux summaries.
- NOAA SWPC event reports.

### Temporal Coverage For 2011-2025

- LASCO/CDAW covers the project period, subject to data gaps.
- DONKI covers the project period but completeness may vary by event type and
  operational status.
- GOES flare data covers the full project period.

### Implementation Difficulty

Medium. Event catalogs need careful time matching, duplicate handling, and
Earth-directed classification.

### Leakage Risk

High if using arrival-time annotations, geomagnetic response, or post-event
classification that was not known at origin `t`. Use only CME/flare observations
available before origin, and avoid columns derived from later in-situ impact.

### Recommended First Features

- Count of CMEs observed in previous 24, 48, and 72 hours.
- Maximum CME plane-of-sky speed in previous 72 hours.
- Maximum CME angular width in previous 72 hours.
- Halo CME flag in previous 72 hours.
- Earth-directed or glancing-blow flag, if based only on coronagraph/early model
  information available before origin.
- Maximum GOES flare class in previous 24, 48, and 72 hours.
- Count of M/X flares in previous 72 hours.

## 4. Derived OMNI Plasma / Regime Features

### Physical Rationale

The current CSV already includes speed, density, temperature, and magnetic-field
magnitude. Derived plasma quantities can expose regime information: compressed
streams, rarefaction regions, hot fast wind, ICME-like intervals, or slow
background wind.

### Expected Effect On 72h Speed Prediction

These features can improve persistence adjustment, regime classification, and
trend interpretation, especially when the recent solar wind at origin is already
in a stream interface or transient disturbance.

### Possible Data Sources

- Existing hourly OMNI data already used in the project.
- Additional OMNI columns if available: vector magnetic field, proton
  temperature, flow pressure, beta, alpha/proton ratio.

### Temporal Coverage For 2011-2025

OMNI has long historical coverage and should cover the project period, with
occasional missing values.

### Implementation Difficulty

Easy to medium. Some quantities can be derived from existing columns; others
need additional OMNI variables.

### Leakage Risk

Low if derived strictly from values at or before origin `t`. Do not use values
between `t` and `t + 72h`.

### Recommended First Features

- Dynamic pressure proxy: `Density * Speed^2`.
- Alfven speed proxy: `B / sqrt(Density)`.
- Alfven Mach number proxy: `Speed / Alfven_speed`.
- Proton entropy proxy: `Temperature / Density^(2/3)`.
- Expected temperature from speed, and ratio `T / T_expected(V)`.
- Flags/scores for likely corotating interaction region, slow wind, fast wind,
  and transient-like hot/low-beta intervals using only lagged values.

## 5. WSA-Enlil Model Output

### Physical Rationale

WSA-Enlil is a physics-based heliospheric forecast model. Even if imperfect, its
Earth-time predicted speed, density, magnetic-field magnitude, and HCS crossings
can be strong external predictors complementary to statistical recurrence and CH
features.

### Expected Effect On 72h Speed Prediction

Potentially high, especially for timing. It can encode magnetic connectivity,
heliospheric propagation, stream interaction, and CME effects if CME runs are
included.

### Possible Data Sources

- NOAA SWPC operational WSA-Enlil forecast archives, if accessible.
- CCMC run archives.
- Locally generated WSA-Enlil or WSA-like model output.

### Temporal Coverage For 2011-2025

Operational archive continuity for the full period is uncertain and must be
verified. Local reruns would be expensive but controllable.

### Implementation Difficulty

Hard unless a ready archive exists.

### Leakage Risk

Very high if using analysis products produced after the target time, reruns with
future boundary conditions, or model outputs initialized after origin `t`. Only
use forecast files issued before or at origin `t`.

### Recommended First Features

- WSA-Enlil predicted speed at Earth for lead times 24, 48, 72, and 96 hours.
- Predicted density and magnetic-field magnitude at Earth for 72h lead.
- Predicted HCS crossing within next 72 hours.
- Predicted stream-interface or shock arrival flag.
- Difference between WSA-Enlil 72h predicted speed and current observed speed.

## 6. Multi-View EUV / STEREO

### Physical Rationale

Earth-view EUV only sees the Earth-facing disk. STEREO can reveal coronal holes
rotating toward the Earth-facing side or far-side structures that will become
geoeffective in a few days.

### Expected Effect On 72h Speed Prediction

Could improve forecasts for CHs not yet optimally visible from Earth, especially
for timing and early detection of large CHs approaching central meridian.

### Possible Data Sources

- STEREO/SECCHI EUVI images.
- STEREO beacon or science data.
- Multi-view CH catalogs if available.

### Temporal Coverage For 2011-2025

STEREO launched before the project period. STEREO-A is available for much of the
period. STEREO-B coverage ends around the 2014 loss-of-contact period, with
limited later recovery. View geometry changes over time, so coverage is not
uniform.

### Implementation Difficulty

Hard. Requires image download, calibration, coordinate transforms, segmentation,
and time-varying spacecraft geometry.

### Leakage Risk

Medium. STEREO observations are causal if timestamped before origin `t`, but
care is needed not to use later Earth-view confirmation or future segmentation.

### Recommended First Features

- STEREO-A CH area in longitude bands rotating toward Earth.
- Estimated Earth-arrival time based on solar rotation from STEREO longitude.
- Multi-view total low-latitude CH area.
- Largest approaching CH area and centroid.
- Agreement/disagreement between Earth-view CH and STEREO-view CH area.

## 7. IPS / Heliospheric Tomography

### Physical Rationale

Interplanetary scintillation observes solar-wind structures in the inner
heliosphere before they reach Earth. Tomographic reconstructions can constrain
stream propagation beyond the solar surface.

### Expected Effect On 72h Speed Prediction

Potentially strong for arrival timing and amplitude, especially when surface CH
features do not capture stream interaction during propagation.

### Possible Data Sources

- ISEE/Nagoya IPS products.
- UCSD IPS tomography products.
- Other IPS solar-wind speed reconstructions if accessible.

### Temporal Coverage For 2011-2025

Coverage and continuity depend on the product. This must be verified before use.

### Implementation Difficulty

Hard. Data access, coordinate mapping, cadence, and tomography geometry are
nontrivial.

### Leakage Risk

High if a tomography product assimilates observations after the forecast origin
or is generated retrospectively over a time window that crosses the target.
Only products available at or before origin `t` are valid.

### Recommended First Features

- IPS-derived speed along near-Earth Parker spiral longitude at origin.
- Tomographic speed at radial shells likely to arrive at Earth within 72h.
- Gradient of heliospheric speed along the Earth-connected trajectory.
- Stream-interface probability from IPS reconstruction.

## Recommended First Three Experiments

### Experiment 1: Derived OMNI Regime Features

This is the fastest next step because it uses existing data. Add lagged and
rolling versions of:

```text
dynamic_pressure_proxy = Density * Speed^2
alfven_speed_proxy = B / sqrt(Density)
alfven_mach_proxy = Speed / alfven_speed_proxy
entropy_proxy = Temperature / Density^(2/3)
temperature_ratio = Temperature / T_expected(Speed)
```

Evaluate with the frozen family:

```text
Direct MLP
ExtraTrees
0.7 MLP + 0.3 ExtraTrees
```

Selection rule:

```text
public fixed 2022-2023 and expanding-window CV only
```

### Experiment 2: F10.7 Solar-Cycle Features

Add daily F10.7 with causal rolling means and trends:

```text
F10.7(t)
rolling_mean_27d
rolling_mean_81d
F10.7(t) - rolling_mean_27d
rolling_mean_27d - rolling_mean_81d
7d trend
27d trend
```

This should be easy to implement and may help with cycle-phase bias.

### Experiment 3: Magnetogram-Enhanced CH Features

Use HMI or GONG magnetograms to add magnetic information to the already useful
representative CH feature set:

```text
mean |Br| inside selected CH windows
signed mean Br inside selected CH windows
area * mean |Br|
dominant polarity
polarity imbalance
```

Start with Earth-facing windows matching the selected CH features. Do not use
full-rotation synoptic maps unless they are causally lagged or built only from
information available at origin `t`.

## General Validation Rules

- Use only features available at or before forecast origin `t`.
- Do not use target-period solar wind, geomagnetic indices, or in-situ response
  as input.
- Fit imputers, scalers, feature selectors, and model hyperparameters on public
  train folds only.
- Select features by public fixed validation and public expanding-window CV.
- Use private 2024-2025 only as a final diagnostic after public selection.
