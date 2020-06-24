```
git clone git@github.com:yaxingwang/UDIT.git
```
You will get new folder whose name is `UDIT` in your current path, then  use `cd UDIT/texture` to enter the downloaded new folder

    

Download [pretrained face](https://drive.google.com/file/d/1VHOgS-NdoVaDCMQTSLOObMFUloaRAp6F/view?usp=sharing) and [texture](https://drive.google.com/file/d/1yzfMmlaMSEa6FQaFbjiO4GGL6snb7Uqb/view?usp=sharing). Adding the data (`black_red_bias`) into path: `UDIT/texture/dataset`, and pretrained face (`layers.npy`) into path: `UDIT/texture/deepface`



Training 
```
python --config configs/UDIT_flat2texture.yaml
```

