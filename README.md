# CubeSat-Diagnostic
## Intro
CubeSat Diagnostic is a project created to help universities and other astronomers alike to analyze data received from their CubeSat satellite and review any anomalies within the telemetry data. It seems many people have created academic papers and have investigated this issue in the past, but there does not seem to be a free, open-source resource where you could investigate CubeSat data to see if there is anything wrong with your project. A significant percentage of CubeSats have total or partial failure, and this project is here to mitigate the ones that fail due to resource imbalance, bugs within code, or other fixable errors once in Orbit.

## What It Does
The user first uploads telemetry data received from their CubeSat into the Streamlit webapp, and the model auto-classifies the columns based on the values within each row of that column. This is done to filter out any columns that are not significant to anomaly investigation, such as datetime or geolocation. The user can override these within the app if needed. Once the user is happy with the columns the model will analyze, it will then go through windowed feature extraction and feed into a Random Forest Model that was trained on the OPSSAT-AD training dataset. The model will give each "anomaly" window a score: The higher the score, the bigger the anomaly. Flagged columns within each window are ranked by contribution strength, so the most anomalous channel appears first with its individual score. The user can then filter through the anomaly windows, filter what column is graphed, and look at what is most important to them.

## Quick Start

**Prerequisites:** Python 3.9+

1. **Clone the repo**
   ```bash
   git clone https://github.com/lucaswaunn/CubeSat-Diagnostic.git
   cd CubeSat-Diagnostic
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Upload your telemetry file** — The app accepts `.csv` or `.xlsx` files. Sample datasets are included in the [data/](data/) directory (e.g., `data/telemetry.csv`, `data/NEPALISAT.xlsx`) if you want to test immediately.

6. **Classify columns and run anomaly detection** — After upload, review the column exclusions and override if necessary, then trigger detection. The pre-trained Random Forest model (`model/model.pkl`) will flag anomalies and score them by severity.

## Evaluation

The `evaluate.py` script validates detections using four independent methods when no ground-truth labels are available: z-score spike detection, Isolation Forest, Mahalanobis Distance, and cross-channel correlation analysis. When labels are present (e.g., OPS-SAT-AD), it computes precision, recall, and F1 instead.

```bash
python evaluate.py data/telemetry.csv
```

Optional flags: `--threshold 0.70`, `--window 500`, `--model model/model.pkl`, `--output evaluation_report.csv`.

## Data Format

The app accepts `.csv` or `.xlsx` files containing numeric telemetry data. Each row should represent a single timestep and each column a telemetry channel (e.g., battery voltage, temperature, gyro readings). The app auto-detects which columns are sensor data and excludes timestamps, counters, and binary status flags — but you can override any classification before running detection. Files should have at least a few hundred rows for the windowing to produce meaningful results. Missing values are handled via `dropna`, but large gaps may affect feature quality.

## Limitations
* Currently there is no temporal context — each window is scored independently, so it can't catch slow drifts or trends that develop over many windows
* The model assumes roughly uniform sampling rate (hardcoded sampling=1), so irregularly sampled or gappy data could produce misleading features
* No retraining pipeline in the app — if someone wants to fine-tune on their own labeled data, they'd need to manually run model.py with their own dataset
* Column classifier uses heuristics and a hardcoded name list tuned to OPS-SAT — other missions' naming conventions might get misclassified

## References

- Ruszczak, B., Kotowski, K., Evans, D., & Nalepa, J. (2025). The OPS-SAT benchmark for detecting anomalies in satellite telemetry. *Scientific Data*. https://doi.org/10.1038/s41597-025-05035-3

- Looney, C., & Wenger, E. (2025). Enhancing CubeSat Telemetry Systems for Autonomous Space Missions Utilizing Machine Learning Techniques. *International Telemetering Conference Proceedings*, 60. http://hdl.handle.net/10150/679583

## Images

**Uploading Data to Site**

<img width="1545" height="955" alt="Uploading data to site" src="https://github.com/user-attachments/assets/0ab94c51-0f94-452d-8edc-e3653693d86f" />

**Anomaly Window Detection**

<img width="1545" height="955" alt="Anomaly window detection" src="https://github.com/user-attachments/assets/ecc3633b-48c0-4b3a-a5d0-35088edb2821" />
