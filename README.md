# Deformable-Tetrahedral-Meshes-with-Gaussian-Splats-for-Fast-Rendering

# Combined Installation Instructions for DefTet and 3D Gaussian Splatting Projects

This README provides a combined guide for setting up both the DefTet and 3D Gaussian Splatting projects.

## Part 1: Installation Instructions for DefTet

### Requirements for DefTet
- Python 3.8
- Pytorch 1.9.0
- CUDA 11.1
- GCC version 6.0 or higher

### Step-by-Step Installation for DefTet
1. **Install Pytorch**:
   ```
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Install QuarTet**:
   Follow the official link: [QuarTet Installation](https://github.com/crawforddoran/quartet).
   ```
   git clone https://github.com/crawforddoran/quartet
   apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev # Dependencies for QuarTet
   cd quartet
   make depend
   make 
   ```

3. **Install Kaolin**:
   Follow instructions on the official page: [Kaolin Installation](https://kaolin.readthedocs.io/en/latest/notes/installation.html).

4. **Install Additional Libraries**:
   ```
   apt-get update
   apt-get install tmux htop -y
   pip install opencv-python tensorboardx meshzoo ipdb imageio
   apt-get install ffmpeg libsm6 libxext6  -y
   HOME_DIR="$PWD"
   cd "$HOME_DIR/utils/lib/tet_adj_share"
   bash do_all.sh
   cd "$HOME_DIR/utils/lib/tet_face_adj"
   bash do_all.sh
   cd "$HOME_DIR/utils/lib/tet_point_adj"
   bash do_all.sh
   cd "$HOME_DIR/utils/lib/colaps_v"
   bash do_all.sh
   cd $HOME_DIR
   ```

## Part 2: Installation Instructions for 3D Gaussian Splatting

### Requirements for 3D Gaussian Splatting
- CUDA-ready GPU with Compute Capability 7.0+
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions (not 11.6)

### Step-by-Step Installation for 3D Gaussian Splatting
1. **Cloning the Repository**:
   Use either SSH or HTTPS method to clone the repository with submodules:
   ```
   # SSH
   git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
   ```
   or
   ```
   # HTTPS
   git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
   ```

2. **Local Setup Using Conda**:
   ```
   SET DISTUTILS_USE_SDK=1 # Windows only
   conda env create --file environment.yml
   conda activate gaussian_splatting
   ```
   Optionally, specify a different package download location and environment path if needed.

3. **Running the Optimizer**:
   Use the `train.py` script with the necessary command line arguments.
   ```
   python train.py -s <path to COLMAP or NeRF Synthetic dataset>
   ```

4. **Evaluation**:
   Follow the provided instructions to evaluate the trained models using `render.py` and `metrics.py`.

5. **Interactive Viewers**:
   Set up and run interactive viewers (`SIBR_remoteGaussian_app` and `SIBR_gaussianViewer_app`) as per the instructions in the README.

6. **Processing Your Own Scenes**:
   Follow the instructions provided for using your images and converting them to suitable formats using `convert.py`.

---

*Note: This guide combines only the installation and setup aspects of both projects. For specific usage, training, and evaluation instructions, refer to the individual project documentation or README files.*

## Training
### Dataset
We use synthetic NeRF dataset.
```bash
cd diff_render/diftet_6_subdiv/6_optim
python optim_with_mask_subdiv_from_gridmov.py --expname <dataset name> --datadir <path to data directory> --savedir <path to output directory> --gaussianpth <path to gaussian point cloud from 3DGS> --remote
```

