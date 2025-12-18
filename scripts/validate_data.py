import os
import sys

REQUIRED_STRUCTURE = {
    "parkinson": ["images"],
    "cxr": ["images"],
    "retina": ["images"],
    "wsi": ["tiles"],
    "audio": ["wav"] # ICBHI extracts to a folder, usually 'ICBHI_final_database'
}

def validate_data(data_dir="data"):
    print(f"Validating data structure in '{data_dir}'...")
    missing = []
    dummy_detected = []
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found.")
        return False

    for module, subpaths in REQUIRED_STRUCTURE.items():
        module_path = os.path.join(data_dir, module)
        if not os.path.exists(module_path):
            missing.append(f"Module dir missing: {module}")
            continue
            
        for sub in subpaths:
            full_path = os.path.join(module_path, sub)
            
            # Special handling for Audio (ICBHI extracts to a specific folder name usually)
            if module == "audio" and sub == "wav":
                # Check if *any* wav files exist in audio/ or audio/wav/ or audio/ICBHI_final_database/
                wavs = []
                for root, dirs, files in os.walk(module_path):
                    for f in files:
                        if f.endswith(".wav"):
                            wavs.append(os.path.join(root, f))
                if not wavs:
                    missing.append(f"Missing .wav files in {module}")
                else:
                    # Check size
                    if os.path.getsize(wavs[0]) < 1000: # < 1KB
                        dummy_detected.append(f"{module} (found dummy wav)")
                continue

            if not os.path.exists(full_path):
                missing.append(f"Missing: {module}/{sub}")
                continue
                
            # Check for real files (not empty dir)
            files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
            if not files:
                 missing.append(f"Empty directory: {module}/{sub}")
            else:
                # Check for dummy files (size check)
                # Dummy images created by PIL are usually small but > 0.
                # Real medical images are usually > 10KB.
                sample_file = os.path.join(full_path, files[0])
                size = os.path.getsize(sample_file)
                if size < 5000: # < 5KB is suspicious for a medical image
                    dummy_detected.append(f"{module} (found small/dummy files)")

    if missing:
        print("\n❌ Data Validation FAILED. Missing components:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease run 'make data_public' to download available public datasets.")
        print("For gated datasets (CXR, Retina, WSI), follow the BYOD instructions.")
        return False
        
    if dummy_detected:
        print("\n⚠️  Data Validation WARNING: Potential dummy data detected.")
        for d in dummy_detected:
            print(f"  - {d}")
        print("Evaluation results will be meaningless with dummy data.")
        # User said "Fail loudly if only the dummy samples are found"
        print("❌ Failing due to dummy data detection. Real data required for production evaluation.")
        return False
    
    print("✅ Data Validation PASSED. Real data detected.")
    return True

if __name__ == "__main__":
    if not validate_data():
        sys.exit(1)
