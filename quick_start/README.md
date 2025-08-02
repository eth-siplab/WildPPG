## Quick Start

We provide **cleaned, ground-truth heart rate data** (using the Pan–Tompkins algorithm) along with corresponding PPG and sensor signal segments for easy experimentation.

* **Structured data is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/eth-siplab/WildPPG/tree/main).**
* **MATLAB users:**
  See [`MATLAB_WildPPG.m`](MATLAB_WildPPG.m) for a ready-to-use script that automatically downloads and loads the dataset from Hugging Face if it is not already present.

---

### Dataset Details

* Each variable is organized as a cell array with one entry per subject (16x1).
* Example signals include PPG (head and wrist), heart rate, altitude, temperature, and IMU from all sensors.

---

