<div align="center">
<h1>[Neuro Computing 2024] Refining and reweighting pseudo labels for weakly supervised object
detection</h1>
</div>

<div align="center">
  <img src="extra/model.png" width="800"/>
</div><br/>

## Notes

This is a Office implementation of the **Refining and reweighting pseudo labels for weakly supervised object** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231224001589).

## Get Started

#### 1. Please follow these steps to create environment.

a. Create a conda virtual environment and activate it.

```shell
conda create -n wsod python=3.7 -y
conda activate wsod
```
b. Install other packets as followings.

- numpy                1.21.6
- opecv-python         4.7.0.72
- torch                1.6.0
- torchvision          0.7.0
- tensorboard          2.11.2


#### 2. Download [BSR_bsds500.tgz](https://drive.google.com/file/d/1gSUgdH1MdPZjGreUa8COnuem5pUTp8iA/view?usp=drive_link) and place it in the main directory.

#### 3. Generate training and testing datasets.

```shell
python create_mnistm.py
```
The script will automatically create a data directory and generate mnist_data_label.hkl, mnist_data.hkl, and mnistm_data.hkl in this directory for training and testing.
```shell
|---data
|   |---mnist_data_label.hkl        # MNIST labels（training + test, one-hot, shared with MNIST-M）
|   |---mnist_data.hkl              # MNIST images（training + test）
|   |---mnistm_data.hkl             # MNIST-M images（traning + test + valid）
```

#### 4. Start Training.

```shell
python model.py
```

**Note that this model is very sensitive to the batch_size, our implementation cannot perform as perfect as the
original paper, so be careful when you tune parameters for other datasets.** 

#### 5. Start Test.


## Result

We only conduct the experiments from mnist to mnist_m, the target accuracy of our implementation is about **85.8%** (original
paper ~83.2%).
