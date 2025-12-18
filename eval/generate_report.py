import json
import os
from fpdf import FPDF

def generate_report():
    print("Generating Investor Report...")
    
    # Load metrics
    metrics = {}
    for service in ["yolo", "nanovlm", "titans"]:
        try:
            with open(f"eval/results/{service}_metrics.json", "r") as f:
                metrics[service] = json.load(f)
        except:
            metrics[service] = {}
            
    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(40, 10, "MedCraft AI - Investor Report")
    pdf.ln(20)
    
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
