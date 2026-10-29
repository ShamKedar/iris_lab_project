import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def main():
    # Load reference and current data
    reference = pd.read_csv("data/reference.csv")
    current = pd.read_csv("data/current.csv")

    # Configure and run Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)

    # Save the visual report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    drift_report.save_html(report_path)
    
    print(f"Evidently Data Drift report successfully saved to {report_path}")

if __name__ == "__main__":
    main()