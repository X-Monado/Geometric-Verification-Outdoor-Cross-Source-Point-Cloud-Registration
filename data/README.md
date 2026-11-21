# Dataset setup

Do not copy the datasets into Git. `configs/paths.local.json` already points to the existing server datasets; it is ignored by Git. Other machines copy `configs/paths.example.json` and provide their own roots.

## KITTI-CrossSource
Official toolkit and download instructions: https://github.com/kezheng1204/KITTI-CrossSource
Official Baidu link: https://pan.baidu.com/s/1I3OTFh6isIA6V-0Mtn2XSg (passcode `1xh4`). The official README currently lists Google Drive as coming soon. Use its downsampled dataset; reconstructing full-resolution data is a separate official-toolkit workflow.

Required tree:
```
KITTI-CrossSource/
  Split/{train,val,test}_cs2.pkl
  downsampled/00/*.npy ... 10/*.npy
  downsampled/00rec/*.npy ... 10rec/*.npy
```
The metadata specifies exact LiDAR/reconstruction pairs and source-to-reference SE(3). If the downloaded archive omits `Split`, obtain the matching metadata from the dataset authors. Evaluation hashes are in `configs/evaluation.json`, and training hashes are in `training/final_stage_protocol.json`. The original [KITTI dataset terms](https://www.cvlibs.net/datasets/kitti/) apply independently of the toolkit's source-code license.

## Cross3DReg
Official dataset project: https://github.com/bestmyselfllll/Cross3DReg
Official Google Drive: https://drive.google.com/file/d/1sEvQQYLJz7reggiM08GXg2Zatck2GLft/view?usp=sharing
Download via the author's link, extract, and point the config at the directory containing metadata and sequences:
```
cross3dreg/
  metadata/{train,val,test}.pkl
  sequences/00/array/*.ply ... 10/array/*.ply
  sequences/00/lidar/*.ply ... 10/lidar/*.ply
```
Images are not consumed. The loader reads binary little-endian float64 XYZ PLY, applies 0.3m centroid voxelization and a deterministic 30000-point cap. The paper split has 7772/1314/4145 train/val/test pairs. KITTI has544 test and168 validation pairs. Follow each dataset's original terms; dataset archives are not redistributed here.

Dataset files stay under your configured data roots. Run `python tools/check_data.py --paths configs/paths.local.json` after extracting them.
