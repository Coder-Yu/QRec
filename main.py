from QRec import QRec
from util.config import ModelConf

if __name__ == '__main__':

    print('='*80)
    print('   QRec: An effective python-based recommendation model library.   ')
    print('='*80)
    # print('0. Analyze the input data.(Configure the visual.conf in config/visual first.)')
    # print('-' * 80)
    print('Generic Recommenders:')
    print('1. UserKNN        2. ItemKNN        3. BasicMF        4. SlopeOne        5. SVD')
    print('6. PMF            7. SVD++          8. EE             9. BPR             10. WRMF')
    print('11. ExpoMF')

    print('MF-based Social Recommenders:')
    print('s1. RSTE          s2. SoRec         s3. SoReg         s4. SocialMF       s5. SBPR')
    print('s6. SREE          s7. LOCABAL       s8. SocialFD      s9. TBPR           s10. SERec')

    print('Network Embedding based Recommenders:')
    print('a1. CoFactor      a2. CUNE-MF       a3. CUNE-BPR      a4. IF-BPR')

    print('DNNs-based Recommenders:')
    print('d1. APR           d2. CDAE          d3. DMF           d4. NeuMF           d5. CFGAN')
    print('d6. IRGAN         d7. RSGAN')

    print('GNNs-based Recommenders:')
    print('g1. NGCF          g2. LightGCN        g3. ESRF        g4. DHCF          g5. DiffNet')

    print('Self-Supervised Recommenders:')
    print('q1. SGL           q2. SEPT            q3. BUIR        q4. MHCN')

    print('Basic Methods:')
    print('b1. UserMean      b2. ItemMean      b3. MostPopular   b4. Rand')

    print('='*80)
    num = input('please enter the number of the model you want to run:')
    import time
    s = time.time()
    #Register your model here and add the conf file into the config directory
    models = {'1':'UserKNN','2':'ItemKNN','3':'BasicMF','4':'SlopeOne','5':'SVD','6':'PMF',
                 '7':'SVD++','8':'EE','9':'BPR','10':'WRMF','11':'ExpoMF',
                 's1':'RSTE','s2':'SoRec','s3':'SoReg','s4':'SocialMF','s5':'SBPR','s6':'SREE',
                 's7':'LOCABAL','s8':'SocialFD','s9':'TBPR','s10':'SEREC','a1':'CoFactor',
                 'a2':'CUNE_MF','a3':'CUNE_BPR','a4':'IF_BPR',
                 'd1':'APR','d2':'CDAE','d3':'DMF','d4':'NeuMF','d5':'CFGAN','d6':'IRGAN','d7':'RSGAN',
                 'g1':'NGCF', 'g2':'LightGCN', 'g3':'ESRF', 'g4':'DHCF', 'g5':'DiffNet',
                 'q1':'SGL', 'q2':'SEPT', 'q3':'BUIR', 'q4':'MHCN',
                 'b1':'UserMean','b2':'ItemMean','b3':'MostPopular','b4':'Rand'}
    try:
        conf = ModelConf('./config/' + models[num] + '.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = QRec(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
