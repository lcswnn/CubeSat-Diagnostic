# CubeSat-Diagnostic
## Intro
CubeSat Diagnostic is a project created to help universities and other astronomers alike to analyze data recived from their CubeSat satellite and review any anomolies within the telemetry data. It seems many people have created academic papers and have investigated this issue in the past, but there does not seem to be a free, open-source resource where you could investigate CubeSat data to see if there is anything wrong with your project. ~41% of CubeSats have total or partial failure, and this project is here to mitigate the ones that fail due to resource inbalance, bugs within code, or other fixable errors once in Orbit.

## What It Does
The user first uploads telemetry data recieved from their CubeSat into the Streamlit webapp, and the model auto-classifies the columns based on the values within each row of that column. This is done to filter out any columns that are not significant to anomaly investigation, such as datetime or geolocation. The user can override these within the app if needed. Once the user is happy with the columns the model will analyze, it will then go through windowed feature extraction and feed into a Random Forest Model that was trained on the OPSSAT-AD training dataset. The model will give each "anomaly" window a score: The higher the score, the bigger the anomaly. The user can then filter through the anomaly windows, filter what column is graphed, and look at what is most important to them.

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

6. **Classify columns and run anomaly detection** — After upload, label each column (temperature, voltage, etc.), then trigger detection. The pre-trained Isolation Forest model (`model/model.pkl`) will flag anomalies and score them by severity.

## Limitations
* Currently there is No temporal context — each window is scored independently, so it can't catch slow drifts or trends that develop over many windows
* The model assumes roughly uniform sampling rate (hardcoded sampling=1), so irregularly sampled or gappy data could produce misleading features
* No retraining pipeline in the app — if someone wants to fine-tune on their own labeled data, they'd need to manually run model.py with their own dataset
* Column classifier uses heuristics and a hardcoded name list tuned to OPS-SAT — other missions' naming conventions might get misclassified

## References

- Ruszczak, B., Kotowski, K., Evans, D., & Nalepa, J. (2025). The OPS-SAT benchmark for detecting anomalies in satellite telemetry. *Scientific Data*. https://doi.org/10.1038/s41597-025-05035-3

- Looney, C., & Wenger, E. (2025). Enhancing CubeSat Telemetry Systems for Autonomous Space Missions Utilizing Machine Learning Techniques. *International Telemetering Conference Proceedings*, 60. http://hdl.handle.net/10150/679583

## Images

<img width="1545" height="955" alt="Screenshot 2026-04-07 at 9 50 17 PM" src="https://github.com/user-attachments/assets/0ab94c51-0f94-452d-8edc-e3653693d86f" />

<img width="1545" height="955" alt="Screenshot 2026-04-07 at 9 50 52 PM" src="https://github.com/user-attachments/assets/ecc3633b-48c0-4b3a-a5d0-35088edb2821" />




