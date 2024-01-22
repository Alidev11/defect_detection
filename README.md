<p align="center"><img src="https://socialify.git.ci/Alidev11/defect_detection/image?description=1&amp;descriptionEditable=Optimization%20of%20Defect%20Detection%20during%20production%20using%20AI&amp;font=Raleway&amp;forks=1&amp;issues=1&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Brick%20Wall&amp;pulls=1&amp;stargazers=1&amp;theme=Dark" alt="project-image" width="850"></p>

<p id="description">GUI application that detects defects in sardine cans during production. The deep learning model used can detect defects in more than just my dataset.</p>
<h2>Project Screenshots:</h2>
<p float="left">
<img src="Rapport-img/Sign_in.png" width="220"> <img src="Rapport-img/add-warehouse.png" width="220">  <img src="Rapport-img/add-product.png" width="220">  
<img src="Rapport-img/show-warehouse.png" width="220">  
<img src="Rapport-img/show-products.png" width="220">  
<img src="Rapport-img/acceuil.png" width="220">  
<img src="Rapport-img/show-gallerie.png" width="220">  
<img src="Rapport-img/image-uploaded.png" width="220">  
<img src="Rapport-img/free-space-localisation.png" width="220">  
</p>
 
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Import or take a picture of cans from 7 sides
*   Classify the can (Good/Bad)
*   Detect defects if the can is bad
*   Display heatmap

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   PyQt5
*   Pytorch
*   Opencv
*   Scikit-learn
*   Numpy
*   Matplotlib
*   Seaborn



## Architecture

**Training.**
VGG16 feature extractor pre-trained on ImageNet, classification head - Average Global Pooling and a Dense layer. Model outputs 2-dimensional vector that contains probabilities for class 'Good' and class 'Anomaly'. Finetuned only last 3 convolutional layers and a dense layer. Loss is Cross-Entropy; optimizer is Adam with a learning rate of 0.0001.


*Model Training Pipeline:*
![model_train_pipeline](docs/model_train_pipeline.png)

**Inference.**
During inference model outputs probabilities as well as the heatmap. Heatmap is the linear combination of feature maps from layer conv5-3 weighted by weights of the last dense layer, and upsampled to match image size. From the dense layer, we take only weights that were used to calculate the score for class 'defective'.

*Model Inference:*
![model_inference](classified/zoo0.png)
