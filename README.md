# {Title}
**Abstract**: Land Surface Temperature (LST) is a key metric for heat island mitigation and cool urban planning meant to deter the effects of climate change for underrepresented areas. This work sets a free and open dataset benchmark for measuring the performance of models to predict a future LST by a monthly basis. According to an extensive literature review search, no other benchmarks exists for monthly temperature at a consistent moderate resolution of 30m. The dataset was scrapped from all U.S cities above 90 square miles resulting in DEM, Land Cover, NDBI, NDVI, NDWI, NDBI & LST for 103 cities. Metrics for temperature prediction include LST and a heat index 1-25 to generalize to individual cities. A baseline measurement was taken with the transformer architecture at 2.6 RMSE of a 1-25 Heat Index and 9.71F RMSE for LST prediction for all 103 cities in the United States. Surface temperature can be effectively predicted and generalized using only a few key variables. SOTA vision architecture, the choice of data and data augmentation contribute to effective pixel-wise prediction.

<div align="center">
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
