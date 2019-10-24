<h1>RecQ</h1>
<h2>Introduction</h2>

**Founder**: [@Coder-Yu ](https://github.com/Coder-Yu)<br>
**Main Contributors**: [@DouTong](https://github.com/DouTong) [@Niki666](https://github.com/Niki666) [@HuXiLiFeng](https://github.com/HuXiLiFeng) [@BigPowerZ](https://github.com/BigPowerZ) [@flyxu](https://github.com/flyxu)<br>

**RecQ** is a Python library for recommender systems (Python 2.7.x) in which a number of the state-of-the-art recommendation models are implemented. To run RecQ easily (no need to setup packages used in RecQ one by one), the leading
 open data science platform  [**Anaconda**](https://www.continuum.io/downloads) is strongly recommended. It integrates Python interpreter, common scientific computing libraries (such as Numpy, Pandas, and Matplotlib), and package manager.
 All of them make it a perfect tool for data science researcher. Besides, GPU based deep models are also available (TensorFlow is required).

<h2>Latest News</h2>
<p>
17/09/2019 - NGCF proposed in SIGIR'19 has been added. </br>
13/08/2019 - RSGAN proposed in ICDM'19 has been added.</br>
09/08/2019 - Our paper is accepted as full research paper by ICDM'19. </br>
02/20/2019 - IRGAN proposed in SIGIR'17 has been added (tuning...) </br>
02/12/2019 - CFGAN proposed in CIKM'18 has been added.</br>
02/04/2019 - NeuMF proposed in WWW'17 has been added.</br>
10/09/2018 - An Adversarial training based Model: APR has been implemented.</br>
10/02/2018 - Two deep models: DMF CDAE have been implemented.</br>
07/12/2018 - Algorithms supported by TensorFlow: BasicMF, PMF, SVD, EE (Implementing...) </br>
</p>

<h2>Architecture of RecQ</h2>

![RecQ Architecture](http://ww3.sinaimg.cn/large/88b98592gw1f9fh8jpencj21d40ouwlf.jpg)

<h2>Features</h2>
<ul>
<li><b>Cross-platform</b>: as a Python software, RecQ can be easily deployed and executed in any platforms, including MS Windows, Linux and Mac OS.</li>
<li><b>Fast execution</b>: RecQ is based on the fast scientific computing libraries such as Numpy and some light common data structures, which make it run much faster than other libraries based on Python.</li>
<li><b>Easy configuration</b>: RecQ configs recommenders using a configuration file.</li>
<li><b>Easy expansion</b>: RecQ provides a set of well-designed recommendation interfaces by which new algorithms can be easily implemented.</li>
<li><b><font color="red">Data visualization</font></b>: RecQ can help visualize the input dataset without running any algorithm. </li>
</ul>

<h2>How to Run it</h2>
<ul>
<li>1.Configure the **xx.conf** file in the directory named config. (xx is the name of the algorithm you want to run)</li>
<li>2.Run the **main.py** in the project, and then input following the prompt.</li>
</ul>
<h2>How to Configure it</h2>
<h3>Essential Options</h3>
<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th width="12%" scope="col"> Entry</th>
    <th width="16%" class="conf" scope="col">Example</th>
    <th width="72%" class="conf" scope="col">Description</th>
  </tr>
  <tr>
    <td>ratings</td>
    <td>D:/MovieLens/100K.txt</td>
    <td>Set the path to input dataset. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
 <tr>
    <td>social</td>
    <td>D:/MovieLens/trusts.txt</td>
    <td>Set the path to input social dataset. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
  <tr>
    <td scope="row">ratings.setup</td>
    <td>-columns 0 1 2</td>
    <td>-columns: (user, item, rating) columns of rating data are used;
      -header: to skip the first head line when reading data<br>
    </td>
  </tr>
  <tr>
    <td scope="row">social.setup</td>
    <td>-columns 0 1 2</td>
    <td>-columns: (trustor, trustee, weight) columns of social data are used;
      -header: to skip the first head line when reading data<br>
    </td>
  </tr>
  <tr>
    <td scope="row">recommender</td>
    <td>UserKNN/ItemKNN/SlopeOne/etc.</td>
    <td>Set the recommender to use. </br>
    </td>
  </tr>
  <tr>
    <td scope="row">evaluation.setup</td>
    <td>-testSet ../dataset/testset.txt</td>
    <td>Main option: -testSet, -ap, -cv </br>
      -testSet path/to/test/file   (need to specify the test set manually)</br>
      -ap ratio   (ap means that the ratings are automatically partitioned into training set and test set, the number is the ratio of test set. e.g. -ap 0.2)</br>
      -cv k   (-cv means cross validation, k is the number of the fold. e.g. -cv 5)</br>
      Secondary option:-b, -p, -cold<br>
      -b val （binarizing the rating values. Ratings equal or greater than val will be changed into 1, and ratings lower than val will be changed into 0. e.g. -b 3.0）</br>
      -p (if this option is added, the cross validation wll be executed parallelly, otherwise executed one by one) </br>
      <b>-tf </b> (model training would be conducted on TensorFlow if TensorFlow has been installed) </br>
      -cold threshold (evaluation on cold-start users, users in training set with ratings more than threshold will be removed from the test set)
     </td>
  </tr>
  <tr>
    <td scope="row">item.ranking</td>
    <td>off -topN -1 </td>
    <td>Main option: whether to do item ranking<br>
      -topN N1,N2,N3...: the length of the recommendation list. *RecQ can generate multiple evaluation results for different N at the same time</br>
    </td>
  </tr>
  <tr>
    <td scope="row">output.setup</td>
    <td>on -dir ./Results/</td>
    <td>Main option: whether to output recommendation results</br>
      -dir path: the directory path of output results.
       </td>
  </tr>  
  </table>
</div>

<h3>Memory-based Options</h3>
<div>
<table class="table table-hover table-bordered">
  <tr>
    <td scope="row">similarity</td>
    <td>pcc/cos</td>
    <td>Set the similarity method to use. Options: PCC, COS;</td>
  </tr>
  <tr>
    <td scope="row">num.shrinkage</td>
    <td>25</td>
    <td>Set the shrinkage parameter to devalue similarity value. -1: to disable simialrity shrinkage. </td>
  </tr>
  <tr>
    <td scope="row">num.neighbors</td>
    <td>30</td>
    <td>Set the number of neighbors used for KNN-based algorithms such as UserKNN, ItemKNN. </td>
  </tr>
  </table>
</div>

<h3>Model-based Options</h3>
<div>
 <table class="table table-hover table-bordered">
  <tr>
    <td scope="row">num.factors</td>
    <td>5/10/20/number</td>
    <td>Set the number of latent factors</td>
  </tr>
  <tr>
    <td scope="row">num.max.iter</td>
    <td>100/200/number</td>
    <td>Set the maximum number of iterations for iterative recommendation algorithms. </td>
  </tr>
  <tr>
    <td scope="row">learnRate</td>
    <td>-init 0.01 -max 1</td>
    <td>-init initial learning rate for iterative recommendation algorithms; <br>
      -max: maximum learning rate (default 1);<br>
    </td>
  </tr>
  <tr>
    <td scope="row">reg.lambda</td>
    <td>-u 0.05 -i 0.05 -b 0.1 -s 0.1</td>
    <td>
      -u: user regularizaiton; -i: item regularization; -b: bias regularizaiton; -s: social regularization</td>
  </tr> 
  </table>
</div>

<h2>How to extend it</h2>
<ul>
<li>1.Make your new algorithm generalize the proper base class.</li>
<li>2.Rewrite some of the following functions as needed.</li>
</ul>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- readConfiguration()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- printAlgorConfig()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- initModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- buildModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- saveModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- loadModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- predict()<br>

<h2>Algorithms Implemented</h2>
<p><b>Note: </b>We use SGD to obtain the local minimum. So, there have some differences between the original
papers and the code in terms of fomula presentation. If you have problems in understanding the code, please open an issue to ask for help. We can guarantee that all the implementations are carefully reviewed and tested. </p>
<p>Any suggestions and criticism are welcomed. We will make efforts to improve RecQ.</p>
<div>

 <table class="table table-hover table-bordered">
  <tr>
		<th>Rating prediction</th>
		<th>Paper</th>
  </tr>
  <tr>
	<td scope="row">SlopeOne</td>
    <td>Lemire and Maclachlan, Slope One Predictors for Online Rating-Based Collaborative Filtering, SDM 2005.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">PMF</td>
    <td>Salakhutdinov and Mnih, Probabilistic Matrix Factorization, NIPS 2008.
     </td>
  </tr> 
  <tr>
    <td scope="row">SoRec</td>
    <td>Ma et al., SoRec: Social Recommendation Using Probabilistic Matrix Factorization, SIGIR 2008.
     </td>
  </tr>
     <tr>
    <td scope="row">SVD++</td>
    <td>Koren, Factorization meets the neighborhood: a multifaceted collaborative filtering model, SIGKDD 2008.
     </td>
  </tr>
    <tr>
    <td scope="row">RSTE</td>
    <td>Ma et al., Learning to Recommend with Social Trust Ensemble, SIGIR 2009.
     </td>
  </tr>
  <tr>
    <td scope="row">SVD</td>
    <td>Y. Koren, Collaborative Filtering with Temporal Dynamics, SIGKDD 2009.
     </td>
  </tr>
  <tr>
    <td scope="row">SocialMF</td>
    <td>Jamali and Ester, A Matrix Factorization Technique with Trust Propagation for Recommendation in Social Networks, RecSys 2010.
     </td>
  </tr>
    <tr>
    <td scope="row">EE</td>
    <td>Khoshneshin et al., Collaborative Filtering via Euclidean Embedding, RecSys2010.
     </td>
  </tr>
    <tr>
    <td scope="row">SoReg</td>
    <td>Ma et al., Recommender systems with social regularization, WSDM 2011.
     </td>
  </tr>
    <tr>
    <td scope="row">LOCABAL</td>
    <td>Tang, Jiliang, et al. Exploiting local and global social context for recommendation, AAAI 2013.
     </td>
  </tr>

  <tr>
    <td scope="row">SREE</td>
    <td>Li et al., Social Recommendation Using Euclidean embedding, IJCNN 2017.
     </td>
  </tr>
    <tr>
    <td scope="row">CUNE-MF</td>
    <td>Zhang et al., Collaborative User Network Embedding for Social Recommender Systems, SDM 2017.
     </td>
  </tr>
    <tr>
    <td scope="row">SocialFD</td>
    <td>Yu et al., A Social Recommender Based on Factorization and Distance Metric Learning, IEEE Access 2017.
     </td>
  </tr>
  </table>

  </br>
  <table class="table table-hover table-bordered">
  <tr>
		<th>Item Ranking</th>
		<th>Paper</th>

   </tr>
  <tr>
	<td scope="row">BPR</td>
    <td>Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">WRMF</td>
    <td>Yifan Hu et al.Collaborative Filtering for Implicit Feedback Datasets, KDD 2009.
     </td>
  </tr>
  <tr>
	<td scope="row">SBPR</td>
    <td>Zhao et al., Leveraing Social Connections to Improve Personalized Ranking for Collaborative Filtering, CIKM 2014<br>
    </td>
  </tr>
  <tr>
	<td scope="row">ExpoMF</td>
    <td>Liang et al., Modeling User Exposure in Recommendation, WWW 2016.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">CoFactor</td>
    <td>Liang et al., Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence, RecSys2016.
     </td>
  </tr>
  <tr>
    <td scope="row">TBPR</td>
    <td>Wang et al. Social Recommendation with Strong and Weak Ties, CIKM 2016.
     </td>
  </tr>
    <tr>
	<td scope="row">CDAE</td>
    <td>Wu et al., Collaborative Denoising Auto-Encoders for Top-N Recommender Systems, WSDM 2016.<br>
    </td>
  </tr>
    <tr>
	<td scope="row">DMF</td>
    <td>Xue et al., Deep Matrix Factorization Models for Recommender Systems, IJCAI 2017.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">NeuMF</td>
    <td>He et al. Neural Collaborative Filtering, WWW 2017.
     </td>
  </tr>
  <tr>
    <td scope="row">CUNE-BPR</td>
    <td>Zhang et al., Collaborative User Network Embedding for Social Recommender Systems, SDM 2017.
     </td>
  </tr>
   <tr>
	<td scope="row">IRGAN</td>
    <td>Wang et al., IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models, SIGIR 2017.<br>
    </td>
  </tr>
    <tr>
	<td scope="row">SERec</td>
    <td>Wang et al., Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation, AAAI 2018.<br>
    </td>
  </tr>
    <tr>
	<td scope="row">APR</td>
    <td>He et al., Adversarial Personalized Ranking for Recommendation, SIGIR 2018.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">IF-BPR</td>
    <td>Yu et al. Adaptive Implicit Friends Identification over Heterogeneous Network for Social Recommendation, CIKM 2018.
     </td>
  </tr>
  <tr>
    <td scope="row">CFGAN</td>
    <td>Chae et al. CFGAN: A Generic Collaborative Filtering Framework based
on Generative Adversarial Networks, CIKM 2018.
     </td>    
    <tr>
    <td scope="row">NGCF</td>
        <td>Wang et al. Neural Graph Collaborative Filtering, SIGIR 2019.
         </td>
      </tr>
    <tr>
    <td scope="row">RSGAN</td>
        <td>Yu et al. Generating Reliable Friends via Adversarial Learning to Improve Social Recommendation, ICDM 2019.
         </td>
      </tr>
  </table>
</div>

<h3>Category</h3>

<table>
<tr>
<th colspan="10" align="left">Generic Recommenders</th>
</tr>
<tr>
<td>UserKNN</td>    <td>ItemKNN</td>    <td>BasicMF   <td>SlopeOne   <td>SVD</td>
<td>PMF</td>        <td>SVD++</td>      <td>EE</td>       <td>BPR</td>        <td>WRMF</td>
</tr>
<tr>
<td>ExpoMF</td>     <td></td>     <td></td>   <td></td>     <td></td>
<td></td>     <td></td>   <td></td>     <td></td> <td></td>
</tr>
<tr>
<th colspan="10" align="left">Social Recommenders</th>
</tr>
<tr>
<td>RSTE</td>    <td>SoRec</td>      <td>SoReg</td>     <td>SocialMF</td>   <td>SBPR</td>
<td>SREE</td>    <td>LOCABAL</td>    <td>SocialFD</td>  <td>TBPR</td>       <td>SERec</td>
</tr>
<tr>
<th colspan="10" align="left">Network Embedding based Recommenders</th>
</tr>
<tr>
<td>CoFactor</td>     <td>CUNE-MF</td>      <td>CUNE-BPR</td>     <td>IF-BPR</td>  <td>  </td>
<td></td>     <td></td>   <td></td>     <td></td>  <td></td>
</tr>
<tr>
<th colspan="10" align="left">Deep Recommenders</th>
</tr>
<tr>
<td>APR</td>     <td>CDAE</td>      <td>DMF</td>       <td>NeuMF</td>      <td>CFGAN</td>
<td>IRGAN</td>     <td></td>   <td></td>     <td></td>  <td></td>
</tr>
<tr>
<th colspan="10" align="left">Baselines</th>
</tr>
<tr>
<td>UserMean</td>      <td>ItemMean</td>     <td>MostPopular</td>     <td>Rand</td> <td>  </td>
<td></td>     <td></td>   <td></td>     <td></td>  <td></td>
</tr>
</table>

<h2>Related Datasets</h2>
<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th rowspan="2" scope="col">Data Set</th>
    <th colspan="5" scope="col" class="text-center">Basic Meta</th>
    <th colspan="3" scope="col" class="text-center">User Context</th> 
    </tr>
  <tr>
    <th class="text-center">Users</th>
    <th class="text-center">Items</th>
    <th colspan="2" class="text-center">Ratings (Scale)</th>
    <th class="text-center">Density</th>
    <th class="text-center">Users</th>
    <th colspan="2" class="text-center">Links (Type)</th>
    </tr> 
  <tr>
    <td><a href="https://pan.baidu.com/s/1qY7Ek0W" target="_blank"><b>Ciao</b></a> [1]</td>
    <td>7,375</td>
    <td>105,114</td>
    <td width="6%">284,086</td>
    <td width="10%">[1, 5]</td>
    <td>0.0365%</td>
    <td width="4%">7,375</td>
    <td width="5%">111,781</td>
    <td>Trust</td>
    </tr> 
  <tr>
    <td><a href="http://www.trustlet.org/downloaded_epinions.html" target="_blank"><b>Epinions</b></a> [2]</td>
    <td>40,163</td>
    <td>139,738</td>
    <td width="6%">664,824</td>
    <td width="10%">[1, 5]</td>
    <td>0.0118%</td>
    <td width="4%">49,289</td>
    <td width="5%">487,183</td>
    <td>Trust</td>
    </tr> 
   <tr>
    <td><a href="https://pan.baidu.com/s/1hrJP6rq" target="_blank"><b>Douban</b></a> [3]</td>
    <td>2,848</td>
    <td>39,586</td>
    <td width="6%">894,887</td>
    <td width="10%">[1, 5]</td>
    <td>0.794%</td>
    <td width="4%">2,848</td>
    <td width="5%">35,770</td>
    <td>Trust</td>
    </tr> 
	 <tr>
    <td><a href="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip" target="_blank"><b>LastFM</b></a> [4]</td>
    <td>1,892</td>
    <td>17,632</td>
    <td width="6%">92,834</td>
    <td width="10%">implicit</td>
    <td>0.27%</td>
    <td width="4%">1,892</td>
    <td width="5%">25,434</td>
    <td>Trust</td>
    </tr> 
  </table>
</div>

<h3>Reference</h3>
<p>[1]. Tang, J., Gao, H., Liu, H.: mtrust:discerning multi-faceted trust in a connected world. In: International Conference on Web Search and Web Data Mining, WSDM 2012, Seattle, Wa, Usa, February. pp. 93–102 (2012)</p>
<p>[2]. Massa, P., Avesani, P.: Trust-aware recommender systems. In: Proceedings of the 2007 ACM conference on Recommender systems. pp. 17–24. ACM (2007) </p>
<p>[3].  G. Zhao, X. Qian, and X. Xie, “User-service rating prediction by exploring social users’ rating behaviors,” IEEE Transactions on Multimedia, vol. 18, no. 3, pp. 496–506, 2016.</p>
<p>[4] Iván Cantador, Peter Brusilovsky, and Tsvi Kuflik. 2011. 2nd Workshop on Information Heterogeneity and Fusion in Recom- mender Systems (HetRec 2011). In Proceedings of the 5th ACM conference on Recommender systems (RecSys 2011). ACM, New York, NY, USA</p>

<h3>Thanks</h3>
If you our project is helpful to you, please cite one of these papers.</br>
@inproceedings{yu2018adaptive,</br>
  title={Adaptive implicit friends identification over heterogeneous network for social recommendation},</br>
  author={Yu, Junliang and Gao, Min and Li, Jundong and Yin, Hongzhi and Liu, Huan},</br>
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},</br>
  pages={357--366},</br>
  year={2018},</br>
  organization={ACM}</br>
}
</br>
</br>
@article{yu2019generating,</br>
  title={Generating Reliable Friends via Adversarial Training to Improve Social Recommendation},</br>
  author={Yu, Junliang and Gao, Min and Yin, Hongzhi and Li, Jundong and Gao, Chongming and Wang, Qinyong},</br>
  journal={arXiv preprint arXiv:1909.03529},</br>
  year={2019}</br>
}
