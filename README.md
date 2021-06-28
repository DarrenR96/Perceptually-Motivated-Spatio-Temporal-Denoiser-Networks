# Readme

The file "Spatial_Temporal Networks.py" contains the network architectures of the spatial and temporal denoiser networks. 
They are created using a subclass of the Keras model. There are also architectures of the ProxIQA and ProxVQA networks in this file. 

After importing the classes, you can create networks from:

```
spatialDenoiserNetwork = SpatialDenoiser().model()
temporalDenoiserNetwork = TemporalDenoiser().model()
proxIQA = ImageQualityAssessment().model()
proxVQA = VideoQualityAssessment().model()
```
