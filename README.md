# RecQ #
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

##How to Run it##
* 1.Configure the **xx.conf** file in the directory named config. (xx is the name of the algorithm you want to run)
* 2.Run the **main.py** in the project, and then input following the prompt.

##How to Configure it##
###Essential Options
 <table class="table table-hover table-bordered">
  <tr>
    <th width="12%" scope="col"> Entry</th>
    <th width="16%" class="conf" scope="col">Example</th>
    <th width="72%" class="conf" scope="col">Description</th>
  </tr>
  <tr>
    <td>ratings</td>
    <td>D:\\MovieLens\\100K.txt<br>
      /home/user/ratings.txt</td>
 
    <td>Set the path to input dataset: &quot;*.wins&quot; for Windows, and &quot;*.lins&quot; for Linux and Unix. It is convenient if you need to frequently switch among different platforms. If not, you can use &quot;dataset.ratings&quot; for short. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
  <tr>
    <td>dataset.social.wins<br>
      dataset.social.lins</td>
    <td>D:\\Epinions\\trust.txt<br>
      /home/user/trust.txt</td>
 
    <td>Set the path to social dataset. Put &quot;-1&quot; to disable it. </td>
  </tr>
  <tr>
    <td scope="row">ratings.setup</td>
    <td>-columns 0 1 2 3 -threshold -1</td>

    <td>-columns: (user, item, [rating, [timestamp]]) columns of rating data are used; -threshold: to convert rating values to binary ones<br>
      --time-unit DAYS, HOURS, MICROSECONDS, MILLISECONDS, MINUTES, NANOSECONDS, [SECONDS]: time unit of timestamps<br>
      --headline: to skip the first head line when reading data<br>
      --as-tensor: to read all columns as a tensor</td>
  </tr>
  <tr>
    <td scope="row">recommender</td>
    <td>RegSVD/SVD++/PMF/etc.</td>

    <td>Set the recommender to use. Available options include: <br>
      Baselines: GlobalAvg, UserAvg, ItemAvg, UserCluster, ItemCluster, Random, Constant, MostPop; <br>
      Extensions:  NMF, SlopeOne, Hybrid, PD, AR, PRankD, External;<br>
      Algorithms: check out the advanced <a href="#algos" class="blue-link page-scroll">algorithms</a> implemented</td>
  </tr>
  <tr>
    <td scope="row">evaluation.setup</td>
    <td>      cv -k 5 -p on -v 0.1 -o on</td>
 
    <td>Main option: test-set; cv; leave-one-out;  given-n;  given-ratio;<br>
      test-set -f path/to/test/file;<br>
      cv -k kfold (default 5); -p on (parallel execution, default), off (singleton, fold-by-fold); <br>
      leave-one-out -t threads (number of threads, used only for target r) -target u, i, r (r by dafault) [--by-date]<br>
      given-n -N number (default 20) -target u, i [--by-date]; 
      given-ratio -r ratio (default 0.8) -target u, i, r [--by-date]<br>
      -target u, i, r: 
      preserve a ratio of ratings relative to users (u), items (i) or ratings (r); --by-date: sort ratings by timestamps<br>
        Commonly optional settings include: <br>
        -v ratio of validation data (derived from training data, default 0)<br>
        -rand-seed N: 
        set the random seed, if not set, system time will be used;<br>
        --test-view all, cold-start, trust-degree min max (default all); 
        <br>
        --early-stop loss, RMSE, MAE, etc: set the criterion for early stop. Note that early-stop may not produce the best performance. </td>
  </tr>
  <tr>
    <td scope="row">item.ranking</td>
    <td>off -topN -1 -ignore -1</td>

    <td>Main option: whether to do item ranking<br>
      -topN: the length of the recommendation list for item recommendation, default -1 for full list; <br>
      -ignore:  the number of the most popular items to ignore; -diverse: whether to use diversity measures</td>
  </tr>
  <tr>
    <td scope="row">output.setup</td>
    <td>on -dir ./Results/ -verbose on</td>

    <td>Main option: whether to output recommendation results<br>
      -dir path: the directory path of output results;<br>
      -verbose on, off: whether to print out intermediate results;<br>
      --save-model: whether to save recommendation model;<br>
      --fold-data: whether to print out traing and test data;<br>
      --measures-only: whether to print other information except measurements; <br>
      --to-clipboard: copy results to clipboard, useful for a single run;<br>
      --to-file filePath: collect results to a specific file, useful for multiple runs, especifially if not all at once. </td>
  </tr>
  <tr>
    <td scope="row">guava.cache.spec</td>
    <td>maximumSize=200</td>

    <td>Set the Guava cache specificaiton, see <a href="http://docs.guava-libraries.googlecode.com/git/javadoc/com/google/common/cache/CacheBuilderSpec.html">more options</a></td>
  </tr>
  <tr>
    <td scope="row">email.setup</td>
    <td><p>on/off -host smtp.gmail.com<br>
      -port 465 -user xxx@yy.com<br>
      -password xxxx<br>
      -auth true<br>
      -to zzz@yy.com

    <td>main option: if email notification is enabled; <br>
      -host: the email server; -port: the port of the email server<br>
      -user: the user name of your email account; <br>
      -password: the password of your email account;<br>
      -auth true/false: whether the email server requires authentification; <br>
      -to:  the email address to which you want to send notification; </td>
  </tr>
  </table>

##How to extend it##
Waiting...