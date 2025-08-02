## Quick Start

We provide **cleaned, ground-truth heart rate data** (using the Pan–Tompkins algorithm) along with corresponding PPG and sensor signal segments for easy experimentation.

* **Structured data is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/eth-siplab/WildPPG/tree/main).**
* **MATLAB:**
  See [`MATLAB_WildPPG.m`](MATLAB_WildPPG.m) for a ready-to-use script that automatically downloads and loads the dataset from Hugging Face if it is not already present.
* **Python:**
  [`python_WildPPG.py`](python_WildPPG.py)

---

### Dataset Details

* Each variable is organized as a cell array with one entry per subject (16x1).
* Signals include PPG, heart rate, altitude, temperature, and IMU from all sensors.

---

