![cassava](images/Forestryimages_cassava.jpg)
Source: [www.wikipedia.org](https://en.wikipedia.org/wiki/Cassava_mosaic_virus#/media/File:Forestryimages_cassava.jpg)
<h1><b>Cassava Disease Classification</b></h1> 

<span style="color: green">Algorithm:<b> Deep Learning - Neural Networks</b> 
</span>

<span style="color:blue"> <b>Table of Content:</b></span>

<ol>
<li><span style="color:grey">Brief Introduction to:</span></li><lu><span style="color:grey">+ Cassava Diseases<br>+ Stakeholders</span></ul>
<li><span style="color:grey">Hypothesis and Model Building</span></li>
<li><span style="color:grey">Exploratory Data Analysis</span></li>
<li><span style="color:grey">Baseline Model Selection</span></li>
<li><span style="color:grey">Model Training and Validation</span></li>

</ol><br>

<h1><b>Cassava Plant</b></h1>

<p style="text-align: justify">As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80 percent of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.</p>

<p style= "text-align: justify">Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.</p>

<h1><b>Cassava Diseases</b></h1>
<h3><b><span style="color: green">Major Diseases:</span></b></h3>
<p style="text-align: justify">There are about four known diseases of cassava plant among them CMD is the most prevalent one:</p>

><ol>
><li><span style="color:grey">CBB: Cassava Bacterial Blight</span></li>
><li><span style="color:grey">CBSD: Cassava Brown Streak Disease</span></li>
><li><span style="color:grey">CGM: Cassava Green Mottle</b></span></li>
><li><span style="color:grey">CMD: Cassava Mosaic Disease</b></span></ol><br>

<h1><b>Stakeholders</b></h1>

<b><span style="color: green">[Our Stakeholder:]("Beautifull_soup.ipynb")</span></b>

<p style="text-align: justify">We selected The Ministry of Agriculture of Uganda as our stakeholder, for it has a direct relationship with the farmers though its extension workers and agricultural experts.</p>

<span style="color:green"><b>Overview of Uganda:</b></span>
 
><p style="text-align: justify"><b>Location and Population:</b> Uganda is a landlocked nation located in East Africa with population about 20 million.</p> 

><p style="text-align: justify"><b>Arable land:</b> Over 25 percent considered suitable for agriculture, which is much higher than the average for sub-Saharan Africa (6.4 percent).</p>

><p style="text-align: justify"><b>GDP:</b> Agriculture accounts for more than 60 percent, 98 percent of export earnings and over 40 percent of government revenue.</p> 

><p style="text-align: justify"><b>Farming and Income:</b> Farming is labour intensive, with women and children providing 60–80 percent of the labour and crops are cultivated both as cash and food security crops.<p>

<h1><b>Bussiness and Data Models</b></h1>

<span style="color: green"><b>Business Model:</b></span>

><b>Target:</b> 
><ol>
><li><span style="color:grey">High yield of Cassava as cash and food crop</span></li>
><li><span style="color:grey">Early detection of disease</span></li></span></ol>

<span style="color: green"><b>Data Model:</b></span>

><b>Target:</b> 
><ol>
><li><span style="color:grey">Min. loss (cost) function</span></li>
><li><span style="color:grey">Lower false Negative</span></li></span></ol>

<b><span style="color:green">From Stakeholder Perspective:</span></b>

><p style="text-align: justify"><b>High false negative:</b> Implies severe impact on the livelihood of these subsistence farmers. It creates a false impression as if the crops are healthy. This will prevent the stakeholders not to take preventive measures prematurely. Disease will spread → famines were happening in the past.</p>

><b>False positive:</b> Too much cassava will be destroyed although they are healthy (loss in income). 

<span style="text-align: justify">To balance the two short comings we will use <b>F-score</b> that is the harmonic mean of precision and recall. Due to its bias as an evaluation metric $F_{1}$ would not a good score to measure accuracy, because recall and precision are evenly weighted.</span>

<p style="text-align: justify">The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0. The beta parameter determines the weight of recall in the combined score. beta less than 1 lends more weight to precision, while beta greater than 1 favors recall (beta near to zero considers only precision, beta near to +inf only recall).</p>

<span style="text-align: justify">The two other commonly used F measures are the $F_{2}$ score, which weights recall higher than precision, and the $F_{0.5}$ score, which puts more emphasis on precision than recall. Since we want to put more emphasis on recall than on precision the $F_{2}$ will be the best metric in our case.</span>

<h1><b>Exploratory Data Analysis</b></h1>

<p style="text-align: justify">In this project, for the classification of cassava leaves as healthy and unhealthy through deep learning classification algorithms, a dataset of <b>21,397</b> labeled images collected during a regular survey in Uganda is introduced. Most images were crowd-sourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala.</p> 

<span style="color: green"><b>Data Distribution:</b></span>

><p style="text-align: justify"><b>Data Imbalance:</b> From our EDA we have observed an imbalance in the dataset, where CMD has 13,158 observations that account for about <b>61.5</b> percent, CBB <b>5.1</b> percent (1,087 observations), CBSD <b>10.2</b> percent (2,189 observations) CGM <b>11.2</b> percent (2,386 observation) and Healthy <b>12.0</b> percent (2,577 observations).</p>

><p style="text-align: justify"><b>Missing Values:</b> From our data analysis we have see that there are no missing values.</p>

><p style="text-align: justify"><b>Image Quality:</b> In our dataset we observed images of poor quality that could have impact in our model prediction. To solve this problem, we employed a lagrangian transformation to filter-out blurry images.</p> 

><p style="text-align: justify"><b>Data Cleanness:</b> We have observed that there are parts of cassava plant and other objects that should not belong to the dataset. This also will have to some extent a negative impact on model accuracy during training.</p>

<h1><b>Baseline Models Selection</b></h1>

<span style="color: green"><b>Baseline Model:</b></span>

>The baseline model is a function of the dot product of the true label and the individual label with the highest distribution in the whole dataset:
$\hat{y} = cmd.y$ (dot product of the scaler and vector quantities)
 
>where $cmd$ is the label with the highest distribution and $y$ is the label in the dataset.

><p style="text-align:justify"><span style="color:green">From our dataset the probability of getting CMD is <b>61.5%</b>, that of CBB <b>5.1%</b>, CBSD <b>10.2%</b>, CGM <b>11.2%</b> and a Healthy one is <b>12.0%</b>. Due to the imbalance in our dataset the <mark>probability distribution function (accuracy)</mark> can not be a good metric for model prediction.</span></p>
