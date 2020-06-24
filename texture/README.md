# Controlling biases and diversity in diverse image-to-image translation 
# Abstract: 
The task of unpaired image-to-image translation is highly challenging due to the lack of explicitcross-domain pairs of instances.  We consider here diverse image translation (DIT), an evenmore challenging setting in which an image can have multiple plausible translations.  This isnormally achieved by explicitly disentangling content and style in the latent representationand sampling different styles codes while maintaining the image content.  Despite the successof current DIT models, they are prone to suffer from bias.  In this paper, we study the problemof bias in image-to-image translation. Biased datasets may add undesired changes (e.g. changegender or race in face images) to the output translations as a consequence of the particularunderlying visual distribution in the target domain.  In order to alleviate the effects of thisproblem we propose the use of semantic constraints that enforce the preservation of desiredimage  properties.   Our  proposed  model  is  a  step  towards  unbiased  diverse  image-to-imagetranslation (UDIT), and results in less unwanted changes in the translated images while stillperforming the wanted transformation.  Experiments on several heavily biased datasets showthe effectiveness of the proposed techniques in different domains such as faces, objects, andscenes.
# Overview 
- [Dependences](#dependences)
- [Installation](#installtion)
- [Instructions](#instructions)
- [Results](#results)
- [References](#references)
- [Contact](#contact)
# Dependences 
- Python2.7, NumPy, SciPy, NVIDIA GPU
- **Tensorflow:** the version should be more 1.0(https://www.tensorflow.org/)
- **Dataset:** bags,faces 

# Installation 
- Install tensorflow
- Opencv 
# Instructions

```
git clone git@github.com:yaxingwang/UDIT.git
```
You will get new folder whose name is 'UDIT' in your current path, then  use 'cd UDIT' to enter the downloaded new folder
    

Download pretrained face(https://drive.google.com/file/d/1VHOgS-NdoVaDCMQTSLOObMFUloaRAp6F/view?usp=sharing) and dataset(https://drive.google.com/file/d/1yzfMmlaMSEa6FQaFbjiO4GGL6snb7Uqb/view?usp=sharing). Adding the data into path: 'UDIT/texture/dataset', and pretrained face (layers.npy) into path: 'UDIT/texture/deepface'



- Training 
```
python --config configs/UDIT_flat2texture.yaml
```

 



# References 
- \[1\] MUNIT(https://arxiv.org/abs/1804.04732) 
# Contact

If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
