# Content and Style Disentanglement for Artistic Style Transfer
***Dmytro Kotovenko, Artsiom Sanakoyeu, Sabine Lang, Björn Ommer*,  ICCV 2019**

[**Project page**](https://compvis.github.io/content-style-disentangled-ST/) | 
[**Paper**](https://compvis.github.io/content-style-disentangled-ST/paper.pdf) | 
[**Supplementary Material**](https://compvis.github.io/content-style-disentangled-ST/Content_and_Style_Disentanglement_for_Artistic_Style_Transfer_ICCV19_supplementary.pdf)

**This is a working version of the code. Updates are coming. Pretrained models too.**

### Requirements
- python 3.6
- tensorflow 1.13.1
- PIL, numpy, scipy
- tqdm



### Training 
1) Download Places365 to use it as a content dataset: [Places365-Standard high-res train mages (105GB)](http://data.csail.mit.edu/places/places365/train_large_places365standard.tar).  
2) Download a couple of artists images from our group [storage](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/NcJj2oLBTYuT1tf).
We recommend taking first collection of Cezanne and Van Gogh images. 
Place them in folders `./data/art_images/cezanne_vangogh/cezanne` and 
`./data/art_images/cezanne_vangogh/vangogh` respectively.
3) To start training run in terminal `bash launch_multistep_training.sh`. 
This script starts training with default parameters for loss weighting and default 
learning schedule. You may want to change them depending on the data you are using.
For parameters description see main.py file.
      


## Our previous works on Style Transfer
- [A Content Transformation Block For Image Style Transfer (CVPR19)](https://github.com/CompVis/content-targeted-style-transfer)
- [A Style-Aware Content Loss for Real-time HD Style Transfer (ECCV18)](https://github.com/CompVis/adaptive-style-transfer)

## Reference
If you use our work in your research, please cite us using the following BibTeX entry.
```
@conference{kotovenko2019iccv,
  title={Content and Style Disentanglement for Artistic Style Transfer},
  author={Kotovenko, Dmytro, and Sanakoyeu, Artsiom, and Lang, Sabine, and Ommer, Bj\"orn},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
