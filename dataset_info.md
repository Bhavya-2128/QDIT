Listed directory paper
Ran command: `python3 -c "
import pypdf
print('testing')
" 2>/dev/null || which pdftotext`
Ran command: `mkdir -p /home/bhavya_2128/.gemini/antigravity-ide/brain/b233e75d-0642-47cb-9390-3581cb595129/scratch
for f in /home/bhavya_2128/QDIT/paper/*.pdf; do
    fname=$(basename "$f")
    pdftotext "$f" "/home/bhavya_2128/.gemini/antigravity-ide/brain/b233e75d-0642-47cb-9390-3581cb595129/scratch/${fname}.txt"
done
ls -lh /home/bhavya_2128/.gemini/antigravity-ide/brain/b233e75d-0642-47cb-9390-3581cb595129/scratch/*.txt
`
Ran command: `python3 -c '
import glob, os, re

txt_files = glob.glob("/home/bhavya_2128/.gemini/antigravity-ide/brain/b233e75d-0642-47cb-9390-3581cb595129/scratch/*.txt")

for tf in txt_files:
    fname = os.path.basename(tf)
    print("="*80)
    print("FILE:", fname)
    with open(tf, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    # Print first 20 lines (title, abstract)
    print("--- TITLE / HEADER ---")
    for l in lines[:25]:
        print(l.strip())
        
    print("\n--- DATASET MENTIONS ---")
    for i, l in enumerate(lines):
        if any(w in l.lower() for w in ["dataset", "data set", "aptos", "kaggle", "messidor", "idrid", "oct", "mnist", "cifar", "eyepacs", "drive", "hrf", "ddr", "e-ophtha", "diaretdb"]):
            # print surrounding lines
            start = max(0, i-2)
            end = min(len(lines), i+4)
            # print matching context snippet
            print(f"Line {i}:", l.strip())
'`
Ran command: `python3 -c '
import glob, os, re

txt_files = glob.glob("/home/bhavya_2128/.gemini/antigravity-ide/brain/b233e75d-0642-47cb-9390-3581cb595129/scratch/*.txt")

for tf in sorted(txt_files):
    fname = os.path.basename(tf)
    print("="*80)
    print("ANALYZING FILE:", fname)
    with open(tf, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    # Extract title
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    print("Title Candidates / Header:")
    for l in lines[:10]:
        print("  *", l)
        
    print("\n--- EXTRACTING DATASET SECTIONS & REFERENCES ---")
    # Search for specific sections
    sections = re.findall(r"(?:dataset|data set|experimental setup|materials and methods|benchmark)[\s\S]{1,1500}", text, re.IGNORECASE)
    for s in sections[:4]:
        print("--- SECTION SNIPPET ---")
        print(s[:600])
        print("...")
'`
Ran command: `python3 -c '
with open("/home/bhavya_2128/.gemini/antigravity-ide/brain/b233e75d-0642-47cb-9390-3581cb595129/scratch/IET Quantum Communication - 2025 - Nandy Pal - FedQCNN  A Privacy‐Preserving Federated Quantum Convolutional Neural Network.pdf.txt", "r") as f:
    text = f.read()

import re
# look for dataset section in FedQCNN
sections = re.findall(r"(?:dataset|data set|experiment|simulation)[\s\S]{1,1500}", text, re.IGNORECASE)
for s in sections[4:10]:
    print("--- FEDQCNN SNIPPET ---")
    print(s[:600])
    print("...")
'`

Here are the detailed descriptions and direct download links for all datasets used and benchmarked by the researchers across all 4 research papers in the [`paper/`](file:///home/bhavya_2128/QDIT/paper) folder.

---

### 📄 Paper 1: *Diabetic Retinopathy Detection Using Quantum Transfer Learning*
* **File:** [`paper/2405.01734v1.pdf`](file:///home/bhavya_2128/QDIT/paper/2405.01734v1.pdf) *(arXiv:2405.01734v1)*

#### Primary Dataset Used by Authors
1. **APTOS 2019 Blindness Detection (Kaggle)**
   * **Task:** 5-Class Retinal Diabetic Retinopathy Severity Grading (Stage 0: No DR, Stage 1: Mild, Stage 2: Moderate, Stage 3: Severe, Stage 4: Proliferative).
   * **Size & Preprocessing:** 3,662 retinal fundus images preprocessed using Ben Graham's Gaussian filter normalization and resized to $224 \times 224$ pixels.
   * **Link:** [Kaggle APTOS 2019 Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

#### Benchmark & Comparative Datasets Reviewed in Section 4.1
| Dataset | Description & Size | Official Link |
| :--- | :--- | :--- |
| **Kaggle / EyePACS** | 88,702 high-resolution fundus images (5 stages) | [Kaggle EyePACS Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) |
| **IDRiD** | 516 retinal fundus images ($4288 \times 2848$) with disease grading and lesion masks | [IEEE DataPort IDRiD](https://idrid.grand-challenge.org/) |
| **MESSIDOR** | 1,200 fundus color images ($1440 \times 960$ to $2304 \times 1536$) from 3 French ophthalmology depts | [ADCIS Messidor Project](https://www.adcis.net/en/third-party/messidor/) |
| **E-Ophtha** | Two subsets: E-Ophtha-EX (47 exudates + 35 normal) and E-Ophtha-MA (148 microaneurysms + 233 normal) | [ADCIS E-Ophtha Database](https://www.adcis.net/en/third-party/e-ophtha/) |
| **DIARETDB1** | 89 fundus images ($1500 \times 1152$, 50° FOV) annotated by 4 experts | [DIARETDB1 Project](https://www.it.lut.fi/project/diaretdb1/) |
| **DDR** | 13,673 fundus images (45° FOV) with 5 DR stages and 757 lesion annotations | [DDR GitHub Dataset](https://github.com/nkicsl/DDR-dataset) |
| **DRIVE** | 40 retinal images ($565 \times 584$) for blood vessel segmentation | [DRIVE Grand Challenge](https://drive.grand-challenge.org/) |
| **HRF** | 45 images ($3504 \times 2336$): 15 DR, 15 Healthy, 15 Glaucoma | [HRF High Resolution Fundus](https://www.idiap.ch/en/dataset/hrf) |
| **STARE** | 400 retina images with clinical diagnostic annotations | [STARE Clemson Project](https://cecas.clemson.edu/~ahoover/stare/) |

---

### 📄 Paper 2: *FedQCNN: A Privacy-Preserving Federated Quantum Convolutional Neural Network for Retinal Image Classification*
* **File:** [`paper/IET Quantum Communication - 2025 - Nandy Pal - FedQCNN...pdf`](file:///home/bhavya_2128/QDIT/paper/IET%20Quantum%20Communication%20-%202025%20-%20Nandy%20Pal%20-%20FedQCNN%20%20A%20Privacy%E2%80%90Preserving%20Federated%20Quantum%20Convolutional%20Neural%20Network.pdf) *(IET Quantum Communication, 2025)*

#### Primary Dataset Used by Authors
1. **E-Ophtha Exudates Sub-database (E-Ophtha-EX)**
   * **Task:** Exudate and lesion patch classification for privacy-preserving federated edge devices.
   * **Details:** 47 exudate-affected images ($2544 \times 1696$) and 35 healthy images. Extracted $16 \times 16$ image patches (5,000 affected + 5,000 healthy for training; 500 affected + 500 healthy for testing).
   * **Link:** [ADCIS E-Ophtha](https://www.adcis.net/en/third-party/e-ophtha/)

#### Secondary Benchmark Datasets Mentioned
* **MedMNIST / RetinaMNIST:** 1,600 retinal fundus images for 5-class DR grading ([MedMNIST Official Website](https://medmnist.com/))
* **SRM Medical College Retinal Dataset:** Clinical retina fundus data from SRM Institute, Tamil Nadu.

---

### 📄 Paper 3: *QuantumNet: An Enhanced Diabetic Retinopathy Detection Model Using Classical Deep Learning–Quantum Transfer Learning*
* **File:** [`paper/main.pdf`](file:///home/bhavya_2128/QDIT/paper/main.pdf) *(MethodsX, Elsevier, 2025)*

#### Primary Dataset Used by Authors
1. **APTOS 2019 Blindness Detection Dataset**
   * **Task:** Binary DR screening (DR vs. Non-DR) and multi-stage classification using hybrid Cirq/PyTorch quantum transfer learning.
   * **Size & Resolution:** 3,662 high-resolution fundus images ($584 \times 565$ pixels) preprocessed with CLAHE and ImageNet normalization.
   * **Link:** [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

---

### 📄 Paper 4: *A Novel Diabetic Retinopathy Detection from Fundus Images Using Hybrid Quantum Convolutional Neural Network Models*
* **File:** [`paper/s41598-026-49227-2.pdf`](file:///home/bhavya_2128/QDIT/paper/s41598-026-49227-2.pdf) *(Scientific Reports, Nature, 2026)*

#### Primary Datasets Used by Authors (Trained and Cross-Validated)
1. **Kaggle EyePACS Dataset**
   * **Size:** 35,126 high-resolution fundoscopic images (25,810 Normal, 2,443 Mild, 5,292 Moderate, 873 Severe, 708 Proliferative DR).
   * **Link:** [Kaggle EyePACS Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
2. **IDRiD (Indian Diabetic Retinopathy Image Dataset)**
   * **Size:** 516 retinal fundus images ($4288 \times 2848$ pixels) with 5-stage DR ground truth (134 Normal, 20 Mild, 136 Moderate, 74 Severe, 49 PDR).
   * **Link:** [IDRiD Grand Challenge](https://idrid.grand-challenge.org/)
3. **MESSIDOR Dataset**
   * **Size:** 1,200 retinal fundus images (546 Normal, 153 Mild DR, 247 Moderate DR, 254 Severe DR).
   * **Link:** [Messidor Database ADCIS](https://www.adcis.net/en/third-party/messidor/)