## Installation (headless/CLI)

```bash
# Clone
git clone https://github.com/idtlab/AIDRIN.git
cd AIDRIN

# Set Up the Conda Environment
conda create -n aidrin-env python=3.10 -y
conda activate aidrin-env
python -m pip install -e .

Optional: run a quick smoke test
```bash
aidrin list
aidrin data-quality headless_demos/loan_applications.csv -v
```

### CLI Usage

```bash
# Quick data quality assessment (recommended)
aidrin data-quality /path/to/data.csv
aidrin data-quality /path/to/data.csv -v  # verbose output with timing

# List available metrics
aidrin list

# How to run a single metric
aidrin run <metric_name> -h # provides the required arguments to run the metric

# Examples
aidrin run completeness /path/to/data.csv
aidrin run duplicity /path/to/data.csv -v
aidrin run outliers /path/to/data.csv

# Run with metric-specific options
aidrin run class-imbalance /path/to/data.csv label
aidrin run k-anonymity /path/to/data.csv "age,zipcode,gender"

# Custom metrics & remedies
aidrin add-custom-module my_audit               # scaffolds aidrin/custom_metrics/my_audit.py with metric() and remedy()
aidrin run custom my_audit data.csv metric      # run the custom metric
aidrin run custom my_audit data.csv remedy      # run the custom remedy and save CSV to aidrin/custom_metrics/remedy_data

# Run batch metrics from config (yaml examples provided inside headless_demos)
aidrin batch <path/to/config>
```

**CLI Options:**
- `-v, --verbose` - Show progress and timing for each metric


### Available Metrics

| Category | Metric | Description | Required Args |
|----------|--------|-------------|---------------|
| data_quality | completeness | Column completeness scores | - |
| data_quality | duplicity | Dataset duplicity ratio | - |
| data_quality | outliers | Outlier proportions for numerical columns | - |
| impact_of_data_on_AI | correlations | Categorical and numerical correlation matrices | columns |
| impact_of_data_on_AI | feature-relevance | Feature relevance using Pearson correlation | categorical-columns, numerical-columns, target-column |
| fairness_and_bias | class-imbalance | Class imbalance degree and distribution | target-column |
| fairness_and_bias | statistical-rates | Statistical rates across sensitive groups | target-column, sensitive-attribute-column |
| fairness_and_bias | representation-rate | Representation rate ratios | columns |
| data_governance | k-anonymity | k-anonymity score | quasi-identifiers |
| data_governance | l-diversity | l-diversity score | quasi-identifiers, sensitive-column |
| data_governance | t-closeness | t-closeness score | quasi-identifiers, sensitive-column |
| data_governance | entropy-risk | Entropy risk score | quasi-identifiers |
| data_governance | single-attribute-risk | Single attribute Markov-model risk | id-column, eval-columns |
| data_governance | multiple-attribute-risk | Multiple attribute Markov-model risk | id-column, eval-columns |
| custom | custom metrics | User-defined metric modules (aidrin/custom_metrics) | varies; run via `aidrin run custom <name> <file>` |
| custom | custom remedies | User-defined remedy modules (aidrin/custom_metrics) | run via `aidrin run custom <name> <file> remedy` |
