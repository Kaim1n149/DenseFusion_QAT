模型放在cache中，若想改變使用的模型，須將fuse.py或fuse_RGB.py中import的模型架構改成models內對應的架構。
# Parameters
| DenseNet_cat | DenseNet_cat_half | DenseNet_add | DenseNet_add_half|
|:------------:|:-----------------:|:------------:|:----------------:|
|147985|37129|37265|9417|
# Train
    python train.py
# Test
    python fuse.py --ir {IR PATH} --vi {VI PATH}
    python fuse_RGB.py --ir {IR PATH} --vi {VI PATH}
