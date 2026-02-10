## Datasets
- [Matlab competition](https://ieee-dataport.org/competitions/federated-deep-learning-csi-estimation-massive-mimo-environments)

- [Wireless Intelligence](https://wireless-intelligence.com/#/dataSet?id=2c92185c7e3f1aa4017e3f2c93e00001)

- [Nvidia Sionna](https://nvlabs.github.io/sionna/phy/tutorials/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#GPU-Configuration-and-Imports)

## Papers
- [Accurate Channel Prediction Based on Transformer:Making Mobility Negligible](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9832933)
- [TIPS: Transformer Based Indoor Positioning System Using Both CSI and DoA of WiFi Signal](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9923937)

## Documents
- [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/35596)

## Project: CSI Transformer (5G Toolbox + PyTorch)

This codebase generates CSI data in MATLAB (5G Toolbox) for the 28 GHz, 60 km/h scenario with SRS period 0.625 ms, then trains a Transformer to predict CSI and compares achievable sum-rate between perfect and predicted CSI.

### 1) Generate dataset (MATLAB)
- Open and run [matlab/generate_dataset.m](matlab/generate_dataset.m).
- Output: [data/csi_dataset.mat](data/csi_dataset.mat)

### 2) Train Transformer (Python)
- Install dependencies from [python/requirements.txt](python/requirements.txt).
- Run [python/train_transformer.py](python/train_transformer.py).
- Output: [data/results.npz](data/results.npz)

### 3) Plot sum-rate comparison
- Run [python/plot_results.py](python/plot_results.py).
- Output: [data/sum_rate.png](data/sum_rate.png)