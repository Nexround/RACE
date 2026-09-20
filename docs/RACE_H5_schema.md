# Unified RACE Results HDF5 Schema

This reference describes the HDF5 structure written by
`packages/race/src/race/core/online_accumulator.py` in online and streaming
online modes.

## File layout

```text
race_results.h5
├── meta/
│   ├── schema/
│   ├── run/
│   ├── race_config/
│   └── axes/
│       ├── value_000000/
│       ├── value_000207/
│       └── value_123456/
└── reports/
    ├── axes/
    │   ├── value_000000/
    │   │   └── modules/
    │   │       ├── attn_pre_output/
    │   │       │   └── layer_00/
    │   │       │       ├── posterior_mean
    │   │       │       ├── cam_score
    │   │       │       ├── nig_sum_e
    │   │       │       └── nig_sum_e2
    │   │       └── mlp_pre_down/
    │   │           └── layer_00/...
    │   └── value_000207/
    └── summary/
        └── metrics
```

## Metadata groups

### `meta/schema`

| Attribute | Type | Value |
| --- | --- | --- |
| `version` | string | `"1.0.0"` |
| `target_model_family` | string | `"llm"` for the public pipelines |

### `meta/run`

| Attribute | Type | Notes |
| --- | --- | --- |
| `model_name` | string | Model identifier |
| `model_revision` | string | May be empty |
| `run_id` | string | Run identifier |
| `created_at` | string | Creation timestamp |
| `device` | string | Compute device |
| `total_instances` | integer | Number of recorded instances |

The `json_payload` dataset is a one-element string array. Its value is one of:

```json
{"mode": "online", "compute_device": "..."}
```

```json
{"mode": "online_streaming", "compute_device": "..."}
```

### `meta/race_config`

The group stores these attributes:

- `mu_0` (float)
- `lambda_0` (float)
- `alpha_0` (float)
- `beta_0` (float)
- `gamma` (float)
- `eps` (float)
- `device_requested` (string)
- `device` (string)

### `meta/axes/value_XXXXXX`

Each axis value is stored directly below `meta/axes`.

| Attribute | Type | Notes |
| --- | --- | --- |
| `axis_type` | string | For example, `"predicted_label"` or `"domain"` |
| `display_name` | string | Human-readable value name |
| `raw_value` | JSON string | Original value metadata |
| `member_count` | integer | Number of member indices |
| `instance_count_total` | integer | Optional total instance count |
| `instance_count_effective` | integer | Optional effective instance count |

Each group contains `members`, an `int64` array with shape `(N,)`. It may also
contain `json_payload`, a one-element string array. Common `raw_value` values
include:

```json
{"predicted_label": "207", "predicted_label_id": 207}
```

```json
{"domain": "law", "domain_id": 123456}
```

## Report groups

### `reports/axes/value_XXXXXX`

| Attribute | Type | Notes |
| --- | --- | --- |
| `axis_path` | string | For example, `"axes/value_000207"` |
| `instance_count_total` | integer | Total matching instances |
| `instance_count_effective` | integer | Instances that contributed evidence |
| `error_count` | integer | Failed matching instances |

Module results are stored below these paths:

- `modules/attn_pre_output/layer_XX`
- `modules/mlp_pre_down/layer_XX`

### Layer groups

Each `layer_XX` group has these attributes:

- `feature_dim` (integer)
- `N` (integer), the number of observations accumulated by the NIG posterior
- `mu_0` (float)
- `lambda_0` (float)
- `alpha_0` (float)
- `beta_0` (float)
- `lambda_n` (float)
- `alpha_n` (float)

Each layer contains these gzip-compressed datasets:

- `posterior_mean` (`float32`, shape `(feature_dim,)`)
- `cam_score` (`float32`, shape `(feature_dim,)`)
- `negative_cam_score` (`float32`, shape `(feature_dim,)`)
- `empirical_mean` (`float32`, shape `(feature_dim,)`)
- `empirical_snr` (`float32`, shape `(feature_dim,)`)
- `nig_sum_e` (`float64`, shape `(feature_dim,)`)
- `nig_sum_e2` (`float64`, shape `(feature_dim,)`)

Online results use `cam_score` for positive CAM values and
`negative_cam_score` for negative ablation scores.

If raw activation statistics were collected for the layer, it also contains
`activation_mean` (`float32`, shape `(feature_dim,)`).

### `reports/summary/metrics`

The summary stores these attributes:

- `total_domains` (integer)
- `total_samples` (integer)
- `total_correct` (integer)
- `gamma` (float)

## Axis value identifiers

Integer-compatible keys use the integer value, padded to six digits. For
example, `207` becomes `value_000207`.

String keys use a deterministic value derived from the first six hexadecimal
digits of their MD5 digest:

```python
value_id = int(md5(str(key)).hexdigest()[:6], 16) % 900000 + 100000
value_key = f"value_{value_id:06d}"
```

The resulting identifier is in the range `[100000, 999999]`.

## Read a result file

```python
import h5py
import numpy as np

h5_path = "result.h5"
value_key = "value_000207"

with h5py.File(h5_path, "r") as h5_file:
    model_name = h5_file["meta/run"].attrs["model_name"]
    gamma = h5_file["meta/race_config"].attrs["gamma"]

    meta_value = h5_file[f"meta/axes/{value_key}"]
    members = meta_value["members"][:]

    report = h5_file[f"reports/axes/{value_key}"]
    total_count = int(report.attrs["instance_count_total"])
    effective_count = int(report.attrs["instance_count_effective"])

    layer = report["modules/attn_pre_output/layer_00"]
    posterior_mean = layer["posterior_mean"][:].astype(np.float32)
    cam_score = layer["cam_score"][:].astype(np.float32)

    top_indices = np.argsort(cam_score)[-10:][::-1]
    print(
        model_name,
        gamma,
        effective_count,
        total_count,
        members[:5],
        top_indices[:3],
        posterior_mean[top_indices[:3]],
    )
```

## Write compatible results

Use the helpers in `packages/race/src/race/io/h5_schema.py`:

- `ensure_race_meta`
- `ensure_axis_value`
- `append_axis_members`
- `ensure_reports_group`
- `ensure_metrics_group`

Use `posterior_mean`, `cam_score`, `nig_sum_e`, and `nig_sum_e2` as the layer
dataset names when writing compatible online results.
