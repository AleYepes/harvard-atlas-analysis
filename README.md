# Exploratory analysis of Growthlab's economic data

https://atlas.hks.harvard.edu/explore/treemap

## Install for notebooks + git

1. Install nbstripout to Remove Unnecessary Jupyter Metadata

```
pip install nbstripout
nbstripout --install
```

2. Use nbdime for Better Diffs and Merging

```
pip install nbdime
nbdime config-git --enable
```

To manually compare two notebook versions:

```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

To resolve conflicts interactively:

```
nbmerge notebook.ipynb
```
