# What plots are the most interesting?
# 1) Min max of shape pca components
# 2) gifs of shape pca components
# 3) elbow plot of explained variances in shape pca
# 4) elbow plot of explained variances in trajectory pca
# 5) boxplot comparisons of different parameters. between different groups.
# 6) scatter plot of trajectory space (but it is unclear how to plot it since there are many components,
#   I can try tsne).
# I can also try tsne on shape pca.
from pathlib import Path
import joblib
import numpy as np
import plots

result_dir = Path("may")
pca1 = joblib.load(result_dir.joinpath('pca1.joblib'))
print(np.cumsum(pca1.explained_variance_ratio_))
lsgpa_shapes = np.load(result_dir.joinpath('lsgpa_shapes.npy'))
print(lsgpa_shapes.shape)
pca1_shapes = np.load(result_dir.joinpath('pca1_shapes.npy'))
print(pca1_shapes.shape)
trajectories = np.load(result_dir.joinpath('trajectories.npy'))
print(trajectories.shape)

plots.pca_animation1(lsgpa_shapes.shape, pca1, pca1_shapes, result_dir.joinpath('pca1'))