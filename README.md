# AIDRIN – AI Data Readiness Inspector

**AIDRIN** (AI Data Readiness Inspector) is a lightweight, open-source tool designed to evaluate the readiness of datasets for AI and machine learning workflows. It provides an intuitive web interface to assess dataset quality, completeness, and structure through quantitative metrics.

For installation, usage, and contribution guidelines, please refer to the [AIDRIN documentation](https://aidrin.readthedocs.io/en/latest/).

## Headless Mode

AIDRIN includes a headless mode for programmatic/CLI usage without the web interface. This is useful for automation, CI/CD pipelines, and integration with other tools like REDI.

### CLI Usage

```bash
# Quick data quality assessment (recommended)
aidrin data-quality /path/to/data.csv
aidrin data-quality /path/to/data.csv -v  # verbose output with timing

# List available metrics
aidrin list
aidrin list --category data_quality

# Run a single metric
aidrin run <metric_name> <file_path>

# Examples
aidrin run completeness /path/to/data.csv
aidrin run duplicity /path/to/data.csv -v
aidrin run outliers /path/to/data.csv --no-viz  # strip visualization data

# Run with metric-specific options
aidrin run class_imbalance /path/to/data.csv --target-column label
aidrin run k_anonymity /path/to/data.csv --quasi-identifiers "age,zipcode,gender"

# Run batch metrics from config
aidrin batch config.json -v
```

**CLI Options:**
- `-v, --verbose` - Show progress and timing for each metric

### Python API

```python
from aidrin.headless import run_metric, run_data_quality, list_available_metrics
from aidrin.headless.config import HeadlessConfig

# Quick data quality assessment (recommended)
result = run_data_quality('/path/to/data.csv', verbose=True)
print(result)
# {'completeness': {...}, 'duplicity': {...}, 'outliers': {...}}

# List available metrics
metrics = list_available_metrics()
for m in metrics:
    print(f"{m['name']} ({m['category']}): {m['description']}")

# Run a single metric
result = run_metric('completeness', '/path/to/data.csv', verbose=True, strip_visualizations=True)
print(result)
# {'Completeness scores': {'col1': 1.0, 'col2': 0.95, ...}, 'Overall Completeness': 0.98}

# Run multiple metrics via batch config
config = HeadlessConfig(
    file_path='/path/to/data.csv',
    metrics=['completeness', 'duplicity', 'outliers'],
)
results = run_batch_metrics(config, verbose=True, strip_visualizations=True)
```

### Available Metrics

| Category | Metric | Description | Required Args |
|----------|--------|-------------|---------------|
| data_quality | completeness | Column completeness scores | - |
| data_quality | duplicity | Dataset duplicity ratio | - |
| data_quality | outliers | Outlier proportions for numerical columns | - |
| impact_of_data_on_AI | correlations | Categorical and numerical correlation matrices | columns |
| impact_of_data_on_AI | feature_relevance | Feature relevance using Pearson correlation | cat_columns, num_columns, target_column |
| fairness_and_bias | class_imbalance | Class imbalance degree and distribution | target_column |
| fairness_and_bias | statistical_rates | Statistical rates across sensitive groups | y_true_column, sensitive_attribute_column |
| fairness_and_bias | representation_rate | Representation rate ratios | columns |
| data_governance | k_anonymity | k-anonymity score | quasi_identifiers |
| data_governance | l_diversity | l-diversity score | quasi_identifiers, sensitive_column |
| data_governance | t_closeness | t-closeness score | quasi_identifiers, sensitive_column |
| data_governance | entropy_risk | Entropy risk score | quasi_identifiers |
| data_governance | single_attribute_risk | Single attribute Markov-model risk | id_column, eval_columns |
| data_governance | multiple_attribute_risk | Multiple attribute Markov-model risk | id_column, eval_columns |

### Supported File Types

- CSV (`.csv`)
- Excel (`.xls`, `.xlsx`, `.xlsm`, `.xlsb`)
- Parquet (`.parquet`)
- JSON (`.json`)
- HDF5 (`.h5`, `.hdf5`)
- NumPy (`.npy`, `.npz`)

### Integration with REDI

AIDRIN headless mode can be used to validate REDI pipeline outputs:

```bash
# Assess data quality before and after REDI processing
aidrin data-quality input_data.csv -v
# Run REDI: redi run input_data.csv
aidrin data-quality output/train.npz -v
```

```python
from aidrin.headless import run_data_quality

# Assess input data quality before REDI processing
input_result = run_data_quality('input_data.csv', verbose=True)
print(f"Input completeness: {input_result['completeness']['Overall Completeness']}")

# Run REDI pipeline...

# Assess output data quality after REDI processing
output_result = run_data_quality('output/train.npz', verbose=True)
print(f"Output completeness: {output_result['completeness']['Overall Completeness']}")
```
## Google Summer of Code (GSoC) Contribution Disclaimer

Contributions to AIDRIN are welcome and encouraged. However, please note that making contributions prior to or during programs such as GSoC **does not guarantee selection to work on this project under GSoC**.

All contributor selections are made independently based on program-specific criteria and evaluation processes.

Please read our [Code of Conduct](./CODE_OF_CONDUCT.md) before contributing.
