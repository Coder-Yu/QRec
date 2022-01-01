<img src="https://raw.githubusercontent.com/Coder-Yu/QRec/master/logo.png" alt="logo" width="400" border="0"><br>
<p float="left"><img src="https://img.shields.io/badge/python-v3.7+-red"> <img src="https://img.shields.io/badge/tensorflow-v1.14+-blue"> <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Coder-Yu/QRec"></p>
<h2>Introduction</h2>

**QRec** is a Python framework for recommender systems (Supported by Python 3.7.4 and Tensorflow 1.14+) in which a number of influential and newly state-of-the-art recommendation models are implemented. QRec has a lightweight architecture and provides user-friendly interfaces. It can facilitate model implementation and evaluation.
<br>
**Founder and principal contributor**: [@Coder-Yu ](https://github.com/Coder-Yu)<br>
**Other contributors**: [@DouTong](https://github.com/DouTong) [@Niki666](https://github.com/Niki666) [@HuXiLiFeng](https://github.com/HuXiLiFeng) [@BigPowerZ](https://github.com/BigPowerZ) [@flyxu](https://github.com/flyxu)<br>
**Supported by**: [@AIhongzhi](https://github.com/AIhongzhi) (<a href="https://sites.google.com/view/hongzhi-yin/home">A/Prof. Hongzhi Yin</a>, UQ), [@mingaoo](https://github.com/mingaoo) (<a href="http://www.cse.cqu.edu.cn/info/2096/3497.htm">A/Prof. Min Gao</a>, CQU) <br> 

<h2>What's New</h2>
<p>
12/10/2021 - BUIR proposed in SIGIR'21 paper has been added. <br>
30/07/2021 - We have transplanted QRec from py2 to py3. <br>
07/06/2021 - SEPT proposed in our KDD'21 paper has been added. <br>
16/05/2021 - SGL proposed in SIGIR'21 paper has been added. <br>
16/01/2021 - MHCN proposed in our WWW'21 paper has been added.<br>
22/09/2020 - DiffNet proposed in SIGIR'19 has been added. <br>
19/09/2020 - DHCF proposed in KDD'20 has been added. <br>
29/07/2020 - ESRF proposed in my TKDE paper has been added. <br>
23/07/2020 - LightGCN proposed in SIGIR'20 has been added. <br>
17/09/2019 - NGCF proposed in SIGIR'19 has been added. <br>
13/08/2019 - RSGAN proposed in ICDM'19 has been added.<br>
09/08/2019 - Our paper is accepted as full research paper by ICDM'19. <br>
20/02/2019 - IRGAN proposed in SIGIR'17 has been added. <br>
12/02/2019 - CFGAN proposed in CIKM'18 has been added.<br>
</p>

<h2>Architecture</h2>

![QRec Architecture](https://i.ibb.co/zJwLXnb/architecture.png)

<h2>Workflow</h2>

![QRec Architecture](https://i.ibb.co/7W9xTfd/workflow.png)

<h2>Features</h2>
<ul>
<li><b>Cross-platform</b>: QRec can be easily deployed and executed in any platforms, including MS Windows, Linux and Mac OS.</li>
<li><b>Fast execution</b>: QRec is based on Numpy, Tensorflow and some lightweight structures, which make it run fast.</li>
<li><b>Easy configuration</b>: QRec configs recommenders with a configuration file and provides multiple evaluation protocols.</li>
<li><b>Easy expansion</b>: QRec provides a set of well-designed recommendation interfaces by which new algorithms can be easily implemented.</li>
</ul>
<h2>Requirements</h2>
<ul>
<li>gensim==4.1.2</li>
<li>joblib==1.1.0</li>
<li>mkl==2022.0.0</li>
<li>mkl_service==2.4.0</li>
<li>networkx==2.6.2</li>
<li>numba==0.53.1</li>
<li>numpy==1.20.3</li>
<li>scipy==1.6.2</li>
<li>tensorflow==1.14.0</li>
</ul>
<h2>Usage</h2>
<p>There are two ways to run the recommendation models in QRec:</p>
<ul>
<li>1.Configure the xx.conf file in the directory named config. (xx is the name of the model you want to run)</li>
<li>2.Run main.py.</li>
</ul>
<p>Or</p>
<ul>
<li>Follow the codes in snippet.py.</li>
</ul>
        
For more details, we refer you to the [handbook of QRec](https://www.showdoc.com.cn/QRecHelp/7342003725025529).           


<h2>Configuration</h2>
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
    <td>Set the file path of the dataset. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
 <tr>
    <td>social</td>
    <td>D:/MovieLens/trusts.txt</td>
    <td>Set the file path of the social dataset. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
  <tr>
    <td scope="row">ratings.setup</td>
    <td>-columns 0 1 2</td>
    <td>-columns: (user, item, rating) columns of rating data are used.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">social.setup</td>
    <td>-columns 0 1 2</td>
    <td>-columns: (trustor, trustee, weight) columns of social data are used.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">mode.name</td>
    <td>UserKNN/ItemKNN/SlopeOne/etc.</td>
    <td>name of the recommendation model. <br>
    </td>
  </tr>
  <tr>
    <td scope="row">evaluation.setup</td>
    <td>-testSet ../dataset/testset.txt</td>
    <td>Main option: -testSet, -ap, -cv (choose one of them) <br>
      -testSet path/to/test/file (need to specify the test set manually)<br>
      -ap ratio (ap means that the ratings are automatically partitioned into training set and test set, the number is the ratio of the test set. e.g. -ap 0.2)<br>
      -cv k (-cv means cross validation, k is the number of the fold. e.g. -cv 5)<br>
      -predict path/to/user list/file (predict for a given list of users without evaluation; need to mannually specify the user list file (each line presents a user)) <br>
      Secondary option:-b, -p, -cold, -tf, -val (multiple choices) <br>
      <b>-val ratio </b> (model test would be conducted on the validation set which is generated by randomly sampling the training dataset with the given ratio.)<br> 
      -b thres （binarizing the rating values. Ratings equal or greater than thres will be changed into 1, and ratings lower than thres will be left out. e.g. -b 3.0）<br>
      -p (if this option is added, the cross validation wll be executed parallelly, otherwise executed one by one) <br>
      <b>-tf </b> (model training will be conducted on TensorFlow (only applicable and needed for shallow models)) <br>
      -cold thres (evaluation on cold-start users; users in the training set with rated items more than thres will be removed from the test set)
     </td>
  </tr>
  <tr>
    <td scope="row">item.ranking</td>
    <td>off -topN -1 </td>
    <td>Main option: whether to do item ranking<br>
      -topN N1,N2,N3...: the length of the recommendation list. *QRec can generate multiple evaluation results for different N at the same time<br>
    </td>
  </tr>
  <tr>
    <td scope="row">output.setup</td>
    <td>on -dir ./Results/</td>
    <td>Main option: whether to output recommendation results<br>
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
    <td scope="row">num.max.epoch</td>
    <td>100/200/number</td>
    <td>Set the maximum number of epoch for iterative recommendation algorithms. </td>
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

<h2>Implement Your Model</h2>
<ul>
<li>1.Make your new algorithm generalize the proper base class.</li>
<li>2.Reimplement some of the following functions as needed.</li>
</ul>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- readConfiguration()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- printAlgorConfig()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- initModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- trainModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- saveModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- loadModel()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- predictForRanking()<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- predict()<br>
<br>

For more details, we refer you to the [handbook of QRec](https://www.showdoc.com.cn/1526742200869027/7347975167213420).           

<h2>Implemented Algorithms</h2>
<div>

 <table class="table table-hover table-bordered">
  <tr>
		<th>Rating prediction</th>
		<th>Paper</th>
  </tr>
  <tr>
	<td scope="row">SlopeOne</td>
    <td>Lemire and Maclachlan, Slope One Predictors for Online Rating-Based Collaborative Filtering, SDM'05.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">PMF</td>
    <td>Salakhutdinov and Mnih, Probabilistic Matrix Factorization, NIPS'08.
     </td>
  </tr> 
  <tr>
    <td scope="row">SoRec</td>
    <td>Ma et al., SoRec: Social Recommendation Using Probabilistic Matrix Factorization, SIGIR'08.
     </td>
  </tr>
     <tr>
    <td scope="row">SVD++</td>
    <td>Koren, Factorization meets the neighborhood: a multifaceted collaborative filtering model, SIGKDD'08.
     </td>
  </tr>
    <tr>
    <td scope="row">RSTE</td>
    <td>Ma et al., Learning to Recommend with Social Trust Ensemble, SIGIR'09.
     </td>
  </tr>
  <tr>
    <td scope="row">SVD</td>
    <td>Y. Koren, Collaborative Filtering with Temporal Dynamics, SIGKDD'09.
     </td>
  </tr>
  <tr>
    <td scope="row">SocialMF</td>
    <td>Jamali and Ester, A Matrix Factorization Technique with Trust Propagation for Recommendation in Social Networks, RecSys'10.
     </td>
  </tr>
    <tr>
    <td scope="row">EE</td>
    <td>Khoshneshin et al., Collaborative Filtering via Euclidean Embedding, RecSys'10.
     </td>
  </tr>
    <tr>
    <td scope="row">SoReg</td>
    <td>Ma et al., Recommender systems with social regularization, WSDM'11.
     </td>
  </tr>
    <tr>
    <td scope="row">LOCABAL</td>
    <td>Tang, Jiliang, et al. Exploiting local and global social context for recommendation, AAAI'13.
     </td>
  </tr>

  <tr>
    <td scope="row">SREE</td>
    <td>Li et al., Social Recommendation Using Euclidean embedding, IJCNN'17.
     </td>
  </tr>
    <tr>
    <td scope="row">CUNE-MF</td>
    <td>Zhang et al., Collaborative User Network Embedding for Social Recommender Systems, SDM'17.
     </td>
  </table>

  <br>
  <table class="table table-hover table-bordered">
  <tr>
		<th>Item Ranking</th>
		<th>Paper</th>

   </tr>
  <tr>
	<td scope="row">BPR</td>
    <td>Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI'09.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">WRMF</td>
    <td>Yifan Hu et al.Collaborative Filtering for Implicit Feedback Datasets, KDD'09.
     </td>
  </tr>
  <tr>
	<td scope="row">SBPR</td>
    <td>Zhao et al., Leveraing Social Connections to Improve Personalized Ranking for Collaborative Filtering, CIKM'14<br>
    </td>
  </tr>
  <tr>
	<td scope="row">ExpoMF</td>
    <td>Liang et al., Modeling User Exposure in Recommendation, WWW''16.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">CoFactor</td>
    <td>Liang et al., Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence, RecSys'16.
     </td>
  </tr>
  <tr>
    <td scope="row">TBPR</td>
    <td>Wang et al. Social Recommendation with Strong and Weak Ties, CIKM'16'.
     </td>
  </tr>
    <tr>
	<td scope="row">CDAE</td>
    <td>Wu et al., Collaborative Denoising Auto-Encoders for Top-N Recommender Systems, WSDM'16'.<br>
    </td>
  </tr>
    <tr>
	<td scope="row">DMF</td>
    <td>Xue et al., Deep Matrix Factorization Models for Recommender Systems, IJCAI'17'.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">NeuMF</td>
    <td>He et al. Neural Collaborative Filtering, WWW'17.
     </td>
  </tr>
  <tr>
    <td scope="row">CUNE-BPR</td>
    <td>Zhang et al., Collaborative User Network Embedding for Social Recommender Systems, SDM'17'.
     </td>
  </tr>
   <tr>
	<td scope="row">IRGAN</td>
    <td>Wang et al., IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models, SIGIR'17'.<br>
    </td>
  </tr>
    <tr>
	<td scope="row">SERec</td>
    <td>Wang et al., Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation, AAAI'18'.<br>
    </td>
  </tr>
    <tr>
	<td scope="row">APR</td>
    <td>He et al., Adversarial Personalized Ranking for Recommendation, SIGIR'18'.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">IF-BPR</td>
    <td>Yu et al. Adaptive Implicit Friends Identification over Heterogeneous Network for Social Recommendation, CIKM'18'.
     </td>
  </tr>
  <tr>
    <td scope="row">CFGAN</td>
    <td>Chae et al. CFGAN: A Generic Collaborative Filtering Framework based
on Generative Adversarial Networks, CIKM'18.
     </td>    
    <tr>
    <td scope="row">NGCF</td>
        <td>Wang et al. Neural Graph Collaborative Filtering, SIGIR'19'.
         </td>
      </tr>
     <tr>
    <td scope="row">DiffNet</td>
        <td>Wu et al. A Neural Influence Diffusion Model for Social Recommendation, SIGIR'19'.
         </td>
      </tr>
    <tr>
    <td scope="row">RSGAN</td>
        <td>Yu et al. Generating Reliable Friends via Adversarial Learning to Improve Social Recommendation, ICDM'19'.
         </td>
      </tr>
     <tr>
    <td scope="row">LightGCN</td>
        <td>He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, SIGIR'20.
         </td>
      </tr>
     <tr>
    <td scope="row">DHCF</td>
        <td>Ji et al. Dual Channel Hypergraph Collaborative Filtering, KDD'20.
         </td>
      </tr>
     <tr>
    <td scope="row">ESRF</td>
        <td>Yu et al. Enhancing Social Recommendation with Adversarial Graph Convlutional Networks, TKDE'20.
         </td>
      </tr>
      <tr>
    <td scope="row">MHCN</td>
        <td>Yu et al. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation, WWW'21.
         </td>
      </tr>
     <tr>
    <td scope="row">SGL</td>
        <td>Wu et al. Self-supervised Graph Learning for Recommendation, SIGIR'21.
         </td>
      </tr>
    <tr>
    <td scope="row">SEPT</td>
        <td>Yu et al. Socially-Aware Self-supervised Tri-Training for Recommendation, KDD'21.
         </td>
      </tr>
          <tr>
    <td scope="row">BUIR</td>
        <td>Lee et al. Bootstrapping User and Item Representations for One-Class Collaborative Filtering, SIGIR'21.
         </td>
      </tr>
  </table>
</div>


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
    <tr>
    <td><a href="https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=0" target="_blank"><b>Yelp</b></a> [5]</td>
    <td>19,539</td>
    <td>21,266</td>
    <td width="6%">450,884</td>
    <td width="10%">implicit</td>
    <td>0.11%</td>
    <td width="4%">19,539</td>
    <td width="5%">864,157</td>
    <td>Trust</td>
    </tr>
    <tr>
    <td><a href="https://www.dropbox.com/sh/20l0xdjuw0b3lo8/AABBZbRg9hHiN42EHqBSvLpta?dl=0" target="_blank"><b>Amazon-Book</b></a> [6]</td>
    <td>52,463</td>
    <td>91,599</td>
    <td width="6%">2,984,108</td>
    <td width="10%">implicit</td>
    <td>0.11%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>  
  </table>
</div>


<h3>Reference</h3>
<p>[1]. Tang, J., Gao, H., Liu, H.: mtrust:discerning multi-faceted trust in a connected world. In: International Conference on Web Search and Web Data Mining, WSDM 2012, Seattle, Wa, Usa, February. pp. 93–102 (2012)</p>
<p>[2]. Massa, P., Avesani, P.: Trust-aware recommender systems. In: Proceedings of the 2007 ACM conference on Recommender systems. pp. 17–24. ACM (2007) </p>
<p>[3]. G. Zhao, X. Qian, and X. Xie, “User-service rating prediction by exploring social users’ rating behaviors,” IEEE Transactions on Multimedia, vol. 18, no. 3, pp. 496–506, 2016.</p>
<p>[4]. Iván Cantador, Peter Brusilovsky, and Tsvi Kuflik. 2011. 2nd Workshop on Information Heterogeneity and Fusion in Recom- mender Systems (HetRec 2011). In Proceedings of the 5th ACM conference on Recommender systems (RecSys 2011). ACM, New York, NY, USA</p>
<p>[5]. Yu et al. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation, WWW'21.</p>
<p>[6]. He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, SIGIR'20.</p>
<h2>Acknowledgment</h2>
<p>This project is supported by the Responsible Big Data Intelligence Lab (RBDI) at the school of ITEE, University of Queensland, and Chongqing University.</p>

If our project is helpful to you, please cite one of these papers.<br>
@inproceedings{yu2018adaptive,<br>
  title={Adaptive implicit friends identification over heterogeneous network for social recommendation},<br>
  author={Yu, Junliang and Gao, Min and Li, Jundong and Yin, Hongzhi and Liu, Huan},<br>
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},<br>
  pages={357--366},<br>
  year={2018},<br>
  organization={ACM}<br>
}
<br>
<br>
@inproceedings{yu2021self,<br>
  title={Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation},<br>
  author={Yu, Junliang and Yin, Hongzhi and Li, Jundong and Wang, Qinyong and Hung, Nguyen Quoc Viet and Zhang, Xiangliang},<br>
  booktitle={Proceedings of the Web Conference 2021},<br>
  pages={413--424},<br>
  year={2021}<br>
}
