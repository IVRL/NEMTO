# NEMTO


> Code of paper 'Neural Environment Matting for Novel View and Relighting Synthesis of Transparent Objects' (ICCV 2023)

### [Project](https://ivrl.github.io/NEMTO//) | [arXiv](https://arxiv.org/abs/2303.11963) 

Dongqing Wang, Tong Zhang, Sabine Süsstrunk

[![DOI](https://zenodo.org/badge/688858421.svg)](https://zenodo.org/doi/10.5281/zenodo.11094758)

![Figure Abstract](/docs/static/images/teaser.png)

>**Abstract:** We propose NEMTO, the first end-to-end neural rendering pipeline to model 3D transparent objects with complex geometry and unknown indices of refraction. Commonly used appearance modeling such as the Disney BSDF model cannot accurately address this challenging problem due to the complex light paths bending through refractions and the strong dependency of surface appearance on illumination. With 2D images of the transparent object as input, our method is capable of high-quality novel view and relighting synthesis. We leverage implicit Signed Distance Functions (SDF) to model the object geometry and propose a refraction-aware ray bending network to model the effects of light refraction within the object. Our ray bending network is more tolerant to geometric inaccuracies than traditional physically-based methods for rendering transparent objects. We provide extensive evaluations on both synthetic and real-world datasets to demonstrate our high-quality synthesis and the applicability of our method.

If you find this project useful for your research, please cite: 

```
@inproceedings{wang2023nemto,
  title={Nemto: Neural environment matting for novel view and relighting synthesis of transparent objects},
  author={Wang, Dongqing and Zhang, Tong and S{\"u}sstrunk, Sabine},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={317--327},
  year={2023}
}
```


## Installation

We use pytorch 1.12

```
conda create --name nemto python=3.9 pip
conda activate nemto

pip install -r requirements.txt
```

## Project Organization

```
src
|
├──chamfer_distance    
|
├──data_creation
|
├──dtasets
|
├──evaluation
|
├──model
|
├──training      <-Starting Point
|
├──utils

```


## Dataset creation
We provide instructions on how to create synthetic transparent object datasets. Details are included in ``data_creation/data_create.ipynb``.

## Training
Similar to PhySG and IDR, our starting point is in the training folder. 

## Acknowledgement

This project is based on  of the code is based on [IDR](https://github.com/lioryariv/idr) and [PhySG](https://github.com/Kai-46/PhySG). We thank the authors for releasing their code.



