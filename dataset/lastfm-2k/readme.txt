
====================
hetrec2011-lastfm-2k
====================

-------
Version
-------

Version 1.0 (May 2011)

-----------
Description
-----------

    This dataset contains social networking, tagging, and music artist listening information 
    from a set of 2K users from Last.fm online music system.
    http://www.last.fm 

    The dataset is released in the framework of the 2nd International Workshop on 
    Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011) 
    http://ir.ii.uam.es/hetrec2011 
    at the 5th ACM Conference on Recommender Systems (RecSys 2011)
    http://recsys.acm.org/2011 

---------------
Data statistics
---------------

    1892 users
   17632 artists
      
   12717 bi-directional user friend relations, i.e. 25434 (user_i, user_j) pairs
         avg. 13.443 friend relations per user
         
   92834 user-listened artist relations, i.e. tuples [user, artist, listeningCount]
         avg. 49.067 artists most listened by each user
         avg. 5.265 users who listened each artist
            
   11946 tags  
   
  186479 tag assignments (tas), i.e. tuples [user, tag, artist]
         avg. 98.562 tas per user
         avg. 14.891 tas per artist
         avg. 18.930 distinct tags used by each user
         avg. 8.764 distinct tags used for each artist

-----
Files
-----
   	    
   * artists.dat
   
        This file contains information about music artists listened and tagged by the users.
   
   * tags.dat
   
   	This file contains the set of tags available in the dataset.

   * user_artists.dat
   
        This file contains the artists listened by each user.
        
        It also provides a listening count for each [user, artist] pair.

   * user_taggedartists.dat - user_taggedartists-timestamps.dat
   
        These files contain the tag assignments of artists provided by each particular user.
        
        They also contain the timestamps when the tag assignments were done.
   
   * user_friends.dat
   
   	These files contain the friend relations between users in the database.
     
-----------
Data format
-----------

   The data is formatted one entry per line as follows (tab separated, "\t"):

   * artists.dat
   
        id \t name \t url \t pictureURL

        Example:
        707	Metallica	http://www.last.fm/music/Metallica	http://userserve-ak.last.fm/serve/252/7560709.jpg

   * tags.dat
 
        tagID \t tagValue
        1	metal
 
   * user_artists.dat
   
        userID \t artistID \t weight
        2	51	13883
   
   * user_taggedartists.dat
  
        userID \t artistID \t tagID \t day \t month \t year
        2	52	13	1	4	2009  
  
   * user_taggedartists-timestamps.dat

        userID \t artistID \t tagID \t timestamp
        2	52	13	1238536800000

   * user_friends.dat

        userID \t friendID
        2	275

------- 
License
-------

   The users' names and other personal information in Last.fm are not provided in the dataset.

   The data contained in hetrec2011-lastfm-2k.zip is made available for non-commercial use.
   
   Those interested in using the data in a commercial context should contact Last.fm staff:
   http://www.lastfm.com/about/contact

----------------
Acknowledgements
----------------

   This work was supported by the Spanish Ministry of Science and Innovation (TIN2008-06566-C04-02), 
   and the Regional Government of Madrid (S2009TIC-1542).

----------
References
----------

   When using this dataset you should cite:
      - Last.fm website, http://www.lastfm.com

   You may also cite HetRec'11 workshop as follows:

   @inproceedings{Cantador:RecSys2011,
      author = {Cantador, Iv\'{a}n and Brusilovsky, Peter and Kuflik, Tsvi},
      title = {2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)},
      booktitle = {Proceedings of the 5th ACM conference on Recommender systems},
      series = {RecSys 2011},
      year = {2011},
      location = {Chicago, IL, USA},
      publisher = {ACM},
      address = {New York, NY, USA},
      keywords = {information heterogeneity, information integration, recommender systems},
   } 

-------
Credits
-------

   This dataset was built by Ignacio Fernández-Tobías with the collaboration of Iván Cantador and Alejandro Bellogín, 
   members of the Information Retrieval group at Universidad Autonoma de Madrid (http://ir.ii.uam.es)

-------   
Contact
-------

   Iván Cantador, ivan [dot] cantador [at] uam [dot] es
