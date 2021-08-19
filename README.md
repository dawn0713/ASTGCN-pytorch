# ASTGCN

Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN）

This is a **Pytorch** implementation of ASTGCN and MSTCGN. Since the official version only has recent component, this code contains all the components and updates the data processing code.

# Datasets

Process dataset

- on PEMS04 dataset

  ```shell
  python prepareData.py --config configurations/PEMS04_astgcn.conf
  ```

- on PEMS08 dataset

  ```shell
  python prepareData.py --config configurations/PEMS08_astgcn.conf
  ```



# Train and Test

- on PEMS04 dataset

  ```shell
  python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf
  ```

- on PEMS08 dataset

  ```shell
  python train_ASTGCN_r.py --config configurations/PEMS08_astgcn.conf
  ```

  

  



