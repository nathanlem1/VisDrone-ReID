# VisDrone-ReID

This code generates a VisDrone-ReID dataset, multi-class ReID dataset, from VisDrone2019 Multi-Object Tracking 
([VisDrone-MOT](https://github.com/VisDrone/VisDrone-Dataset)) dataset.

Train and Val MOT sequences create the ReID train set, whereas test MOT sequences create the ReID test set.
Every Nth frame of each sequence (N = 5, 10, etc.) is used to crop the object's image patches. When evaluating the ReID
performance, each image crop of the test set is taken as query image once and is compared against all other test images
that serve as gallery. Look at the `prepare_visdrone_reid_dataset.py` for more details.


## Citation

If you find this work useful in your research, please consider citing it using:

```
@misc{VisDrone-ReID2026,
    title={{VisDrone-ReID}: Multi-class ReID dataset prepared from VisDrone-MOT dataset.},
    author={Nathanael L. Baisa},
    howpublished = {\url{https://github.com/nathanlem1/VisDrone-ReID}},
    year={2026}
}
```