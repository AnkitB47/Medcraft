# MedCraft Data Setup

MedCraft requires real medical datasets for meaningful evaluation.

## Automatic Setup (Public Data)
Run the following command to download publicly available datasets (Parkinson HandPD, Audio ICBHI):
```bash
make data_public
```

## BYOD (Bring Your Own Data) Setup
For gated datasets, you must download them manually and place them in the `data/` directory.

### 1. Parkinson (HandPD)
*   **URL**: [Unesp HandPD](http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/)
*   **Action**: Download 'Spiral' and 'Meander' images.
*   **Path**: `data/parkinson/images/`

### 2. CXR (Chest X-Ray)
*   **URL**: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
*   **Action**: Download images and extract.
*   **Path**: `data/cxr/images/`

### 3. Retina (IDRiD)
*   **URL**: [IDRiD Challenge](https://idrid.grand-challenge.org/)
*   **Action**: Register and download.
*   **Path**: `data/retina/images/`

### 4. WSI (Camelyon16)
*   **URL**: [Camelyon17](https://camelyon17.grand-challenge.org/)
*   **Action**: Download slides/tiles.
*   **Path**: `data/wsi/tiles/`

### 5. Audio (ICBHI 2017)
*   **URL**: [ICBHI Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
*   **Action**: Download database zip.
*   **Path**: `data/audio/wav/`

## Validation
Run the following to verify your data setup:
```bash
make data_byod_check
```
**Note**: Evaluation scripts will FAIL if dummy data is detected.
