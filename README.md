# Land Surface Temperature Forecasting Benchmarks and Datasets for Large-Scale Urban Metropolitan Areas
**Abstract**: Land Surface Temperature (LST) serves as a critical indicator for quantifying urban heat islands and informing climate-resilient urban planning strategies, particularly for vulnerable communities. We present a comprehensive open-source benchmark dataset for predicting monthly LST at a spatially consistent resolution of 30m. The dataset encompasses 103 U.S. cities exceeding 90 square miles, providing Digital Elevation Models (DEM), Land Cover classifications, and multiple spectral indices (NDBI, NDVI, NDWI) alongside LST measurements. We implement dual evaluation metrics: LST values and a normalized heat index (1-25) that facilitates cross-city generalization. Our transformer-based baseline achieves 2.6 RMSE for heat index prediction and 9.71F RMSE for LST forecasting across all cities. Notably, we demonstrate that accurate surface temperature prediction can be achieved through a carefully selected subset of geospatial variables when coupled with state-of-the-art vision architectures and appropriate data augmentation techniques.

<div align="center">
  https://jesseguerrero.github.io/LST-Visualization/
  <p float="left">
    <img src="https://i.imgur.com/AGNpQJa.png" width="32%" style="border: 1px solid #ddd; border-radius: 4px; margin: 0 0.5%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);" />
    <img src="https://i.imgur.com/G735JWp.png" width="32%" style="border: 1px solid #ddd; border-radius: 4px; margin: 0 0.5%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);" />
    <img src="https://i.imgur.com/dkjJvBC.png" width="32%" style="border: 1px solid #ddd; border-radius: 4px; margin: 0 0.5%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);" />
  </p>
</div>

<div align="center">
<table>
<thead>
<tr>
<th colspan="5" align="center">1 Month (F)</th>
<th colspan="5" align="center">1 Month (P)</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td>Resnet18</td>
<td>Resnet50</td>
<td>B3</td>
<td>B5</td>
<td></td>
<td>Resnet18</td>
<td>Resnet50</td>
<td>B3</td>
<td>B5</td>
</tr>
<tr>
<td>Unet</td>
<td>20.74</td>
<td>12.47</td>
<td></td>
<td></td>
<td>Unet</td>
<td>5.17</td>
<td>3.47</td>
<td></td>
<td></td>
</tr>
<tr>
<td>DeepLabV3+</td>
<td>21.18</td>
<td>13.34</td>
<td></td>
<td></td>
<td>DeepLabV3+</td>
<td>5.18</td>
<td>3.75</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Segformer</td>
<td></td>
<td></td>
<td>10.04</td>
<td><b>9.71</b></td>
<td>Segformer</td>
<td></td>
<td></td>
<td>2.69</td>
<td><b>2.6</b></td>
</tr>
</tbody>
</table>
<table>
<thead>
<tr>
<th colspan="5" align="center">3 Month (F)</th>
<th colspan="5" align="center">3 Month (P)</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td>Resnet18</td>
<td>Resnet50</td>
<td>B3</td>
<td>B5</td>
<td></td>
<td>Resnet18</td>
<td>Resnet50</td>
<td>B3</td>
<td>B5</td>
</tr>
<tr>
<td>Unet</td>
<td>22.25</td>
<td>22.36</td>
<td></td>
<td></td>
<td>Unet</td>
<td>5.21</td>
<td>5.19</td>
<td></td>
<td></td>
</tr>
<tr>
<td>DeepLabV3+</td>
<td>22.29</td>
<td>22.2</td>
<td></td>
<td></td>
<td>DeepLabV3+</td>
<td>5.21</td>
<td>5.17</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Segformer</td>
<td></td>
<td></td>
<td>11.99</td>
<td>11.8</td>
<td>Segformer</td>
<td></td>
<td></td>
<td>2.88</td>
<td>2.9</td>
</tr>
</tbody>
</table>
</div>

## Links
- Paper: -- 
- Checkpoints: https://bit.ly/4jP7z91
- Data: https://huggingface.co/datasets/JesseGuerrero/LandsatTemperature
- Code: This repository
- Visualization: https://github.com/JesseGuerrero/LST-Visualization

## Usage
Run Landsat imagery into the checkpoints given or train your own model using this repository. Take the inferred Tif and place it into your application.
1. Download checkpoints, use SegFormer 1, 3 month
2. Open inference script
3. Place Landsat imagery
4. Run inference
5. Place combined TIF into application

## Real-world Application Examples
Place the inference tif in any of these frameworks...
- [Esri Maps SDK](https://developers.arcgis.com/javascript/latest/)
- [DeckGL](https://deck.gl/)
- [Cesium](https://cesium.com/)
- [Omniverse](https://www.nvidia.com/en-us/omniverse/)
- [Unity](https://unity.com/)
- Etc.

## Citation
If you use this work please cite our [paper --]():

```bibtex
@article{--,
    author = {--},
    doi = {--},
    journal = {--},
    month = --,
    title = {--},
    url = {--},
    year = {--}
}
```
