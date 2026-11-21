# GitHub publication

Before publishing, add the project license chosen by the authors, confirm the distribution terms for the four checkpoints, and add the author and paper citation once their metadata is final. Keep [THIRD_PARTY_LICENSE.GeoTransformer](../THIRD_PARTY_LICENSE.GeoTransformer) with the vendored code. The code license and checkpoint distribution decision belong to the authors.

Install [Git LFS](https://git-lfs.com/) before staging and review the GitHub LFS terms for the target account. Four checkpoints total about 196 MiB. The package excludes local dataset configuration, raw datasets, compiled objects, logs, and run outputs. The 273 candidate feature archives are kept locally while redistribution scope is unresolved. Downloading the dataset or using the toolkit's source license is separate from permission to release derived artifacts. See the [KITTI dataset terms](https://www.cvlibs.net/datasets/kitti/), [KITTI-CrossSource toolkit](https://github.com/kezheng1204/KITTI-CrossSource) and [Cross3DReg project](https://github.com/bestmyselfllll/Cross3DReg).

After author review, run the staging checks from the project root:

```bash
git lfs install --local
python tools/check_release.py --write-manifest
git add .
python tools/check_release.py --staged
git lfs ls-files
git diff --cached --stat
```

The staged check validates each large-file pointer against its local checkpoint and reports self-generated files that would enter the commit. See [training scope](TRAINING.md) for replay coverage. After publishing, tag the revision used for the manuscript and put its real URL in the availability statement.
