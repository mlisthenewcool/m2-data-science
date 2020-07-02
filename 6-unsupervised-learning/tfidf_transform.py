import time
import numpy as np

"""
    Code tf-idf transform from a set of sample X
    
    Step 1 - Compute idf for each word of the vocabulary over the entire set.
    Step 2 - Compute tf for each sample
    Step 3 - Compute tf times idf to build transformed sample descriptors
"""
def tf_idf_transform(corpus, doc_ids, n_doc, n_voca):
    t = time.time()
    idf = np.zeros(shape=(n_voca,1))
    for i in range(n_doc):
        for j in range(len(doc_ids[i])):
            idf[doc_ids[i][j]] += 1
            #break
    
    idf = np.divide(idf,n_doc)
        
    zezeros = np.where(idf == 0)[0] 
    idf[zezeros] = 0.0000000001
    
    idf = np.divide(1,idf)
    idf = np.log(idf)
    
    elapsed = time.time() - t
    print (elapsed)
    

    t = time.time()
    tf = np.zeros(shape=(n_voca,n_doc))
    for i in range(n_doc):
        #print(i,n_doc)
        for j in range(len(corpus[i])):  
            tf[corpus[i][j]][i] += 1
        tf[:][i] = np.divide(tf[:][i],len(corpus[i]))
        
#    print(n_doc)
#    print(idf.shape)
#    print(tf.shape)

#    for i in range(n_doc):
#        #print(i,n_doc)
#        np.multiply(tf[:][i],idf)
    for i in range(idf.size):
        #print(i,n_doc)
        tf[i][:]=np.multiply(tf[i][:],idf[i])
                
    elapsed = time.time() - t
    print (elapsed)  
            
    return tf
