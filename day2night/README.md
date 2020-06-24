```
git clone git@github.com:yaxingwang/UDIT.git
```
You will get new folder whose name is `UDIT` in your current path, then  use `cd UDIT/day2night` to enter the downloaded new folder

    

Download [pretrained face](https://drive.google.com/file/d/1VHOgS-NdoVaDCMQTSLOObMFUloaRAp6F/view?usp=sharing) and [day_night](https://drive.google.com/file/d/1h4WAWxhbfJvhZOJuNUZM1XAfEgHWJvXV/view?usp=sharing). Adding the data (`day_night`) into path: `cd UDIT/day2night/dataset`, and pretrained face (`layers.npy`) into path: `cd UDIT/day2night/deepface`



Training 
```
python --config configs/UDIT_day2night.yaml
```

