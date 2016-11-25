![RecQ](http://chuantu.biz/t5/43/1480095308x760008152.png)

Released by School of Software Engineering, Chongqing University
##Introduction##
**RecQ** is a Python library for recommender systems (Python 2.7.x). It implements a suit of state-of-the-art recommendations. To run RecQ easily (no need to setup packages used in RecQ one by one), the leading open data science platform  [**Anaconda**](https://www.continuum.io/downloads) is strongly recommended. It integrates Python interpreter, common scientific computing libraries (such as Numpy, Pandas, and Matplotlib), and package manager, all of them make it a perfect tool for data science researcher.
##Architecture of RecQ##

![RecQ Architecture](http://ww3.sinaimg.cn/large/88b98592gw1f9fh8jpencj21d40ouwlf.jpg)

To design it exquisitely, we brought some thoughts from another recommender system library [**LibRec**](https://github.com/guoguibing/librec), which is implemented with Java.

##Features##
* **Cross-platform**: as a Python software, RecQ can be easily deployed and executed in any platforms, including MS Windows, Linux and Mac OS.
* **Fast execution**: RecQ is based on the fast scientific computing libraries such as Numpy and some light common data structures, which make it runs much faster than other libraries based on Python.
* **Easy configuration**: RecQ configs recommenders using a configuration file.
* **Easy expansion**: RecQ provides a set of well-designed recommendation interfaces by which new algorithms can be easily implemented.
* **<font color="red">Data visualization</font>**: RecQ can help visualize the input dataset without running any algorithm. 

##How to Run it##
* 1.Configure the **xx.conf** file in the directory named config. (xx is the name of the algorithm you want to run)
* 2.Run the **main.py** in the project, and then input following the prompt.

##How to Configure it##
###Essential Options
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

    <td>Set the recommender to use. <br>
    </td>
  </tr>
  <tr>
    <td scope="row">evaluation.setup</td>
    <td>../dataset/FilmTrust/testset.txt</td>
 
    <td>Main option: -testSet;<br>
      -testSet -f path/to/test/file;<br>
     </td>
  </tr>
  <tr>
    <td scope="row">item.ranking</td>
    <td>off -topN -1

    <td>Main option: whether to do item ranking<br>
      -topN: the length of the recommendation list for item recommendation, default -1 for full list; <br>
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

###Memory-based Options
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

###Model-based Options
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
##How to extend it##
* 1.Make your new algorithm generalize the proper base class.
* 2.Rewrite some of the following functions as needed.
 - **readConfiguration()**
 - **printAlgorConfig()**
 - **initModel()** 
 - **buildModel()**
 - **saveModel()**
 - **loadModel()**
 - **predict()**
