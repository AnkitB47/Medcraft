import json
import os
from fpdf import FPDF
import datetime
import subprocess

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def generate_report():
    print("Generating Investor Report...")
    
    # Load metrics
    metrics = {}
    required_services = ["yolo", "nanovlm", "titans"]
    
    for service in required_services:
        path = f"eval/results/{service}_metrics.json"
        if not os.path.exists(path):
            print(f"ERROR: Metrics for {service} not found at {path}. Run eval_{service}.py first.")
            # Fail loudly as requested
            exit(1)
            
        with open(path, "r") as f:
            metrics[service] = json.load(f)
            
    # Metadata
    timestamp = datetime.datetime.now().isoformat()
    commit_hash = get_git_revision_short_hash()
    
    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(40, 10, "MedCraft AI - Investor Report")
    pdf.ln(10)
    
    pdf.set_font("Arial", "I", 10)
    pdf.cell(40, 10, f"Generated: {timestamp}")
    pdf.ln(5)
    pdf.cell(40, 10, f"Commit: {commit_hash}")
    pdf.ln(15)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "1. Hybrid Vision (YOLO + ViT)")
    pdf.ln(10)
    pdf.set_font("Arial", "", 10)
    for k, v in metrics.get("yolo", {}).items():
        pdf.cell(40, 10, f"{k}: {v}")
        pdf.ln(6)
        
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "2. NanoVLM (TinyLlama + Perceiver)")
    pdf.ln(10)
    pdf.set_font("Arial", "", 10)
    for k, v in metrics.get("nanovlm", {}).items():
        pdf.cell(40, 10, f"{k}: {v}")
        pdf.ln(6)
        
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "3. Titans Memory (Neural Memory)")
    pdf.ln(10)
    pdf.set_font("Arial", "", 10)
    for k, v in metrics.get("titans", {}).items():
        pdf.cell(40, 10, f"{k}: {v}")
        pdf.ln(6)
        
    pdf.output("eval/INVESTOR_REPORT.pdf")
    
    # Save combined metrics
    with open("eval/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Report generated: eval/INVESTOR_REPORT.pdf")

if __name__ == "__main__":
    generate_report()
