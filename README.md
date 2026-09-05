<p align="center">
<h1 align="center"><strong>OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras</strong></h1>
<h3 align="center">IROS 2026</h3>

<p align="center">
    <a>Yongzhi Lin</a><sup>1*</sup>,</span>
    <a href="https://scholar.google.com/citations?user=ezlbBgUAAAAJ">Kai Luo</a><sup>1*</sup>,
    <a href="https://scholar.google.com/citations?user=_g0m5a0AAAAJ">Yuanfan Zheng</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=0EI9msQAAAAJ">Hao Shi</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=amFy9D8AAAAJ">Mengfei Duan</a><sup>1</sup>,
    <a href="https://github.com/YonderL">Yang Liu</a><sup>1</sup>,
    <a href="https://yangkailun.com">Kailun Yang</a><sup>1†</sup>
    <br>
        <sup>1</sup>Hunan University,
        <sup>2</sup>Zhejiang University
</p>

## OccTrack360 [[PDF]](https://arxiv.org/pdf/2603.08521)
## Installation and Data Preparation for FoSOcc
### Step 1. Installation
Clone FoSOcc
```sh
git clone https://github.com/YouthZest-Lin/OccTrack360
cd OccTrack360
```

Create conda environment

```sh
conda create -n FoSOcc python=3.8
conda activate FoSOcc
# PyTorch 1.12.1 + CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Install other dependencies

```sh
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
pip install yapf==0.40.1
```

Compile CUDA extensions

```sh
pip install -v -e . 
```

### Step 2. Data Preparation

To evaluate our FoSOcc on TrackOcc-Waymo, you can download the data on [Hugging Face](https://huggingface.co/datasets/zgchen33/TrackOcc_waymo). And our [OccTrack360 dataset](https://huggingface.co/datasets/YouthZestLin/OccTrack360) will be coming soon.

The preparation of TrackOcc-Waymo is the same as [TrackOcc](https://github.com/Tsinghua-MARS-Lab/TrackOcc).


## 🎥 Demo
<video src="https://github.com/user-attachments/assets/aea574e5-3a7b-4ba8-b2cd-791202392e5e" controls="controls" muted="muted" autoplay="autoplay" loop="loop" width="100%">
</video>

## 🤝 Publication:
Please consider referencing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```
@inproceedings{lin2026occtrack360,
  title={OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras},
  author={Yongzhi Lin and Kai Luo and Yuanfan Zheng and Hao Shi and Yang Liu and Kailun Yang},
  booktitle={2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```
