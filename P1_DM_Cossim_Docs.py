import nltk,math
import os
from math import log10,sqrt
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()

corpusroot = "C:/Users/dell/presidential_debates"# 30 Documents folder path
tfidfvector={}                          # Vector to store tf-idf values for all documents
doc_feq=Counter()                        # To store document frequency value
tf_all={}                            # To store values of term frequency of all tokens of all docs
lengths_doc=Counter()                   # To calculate length of documents
posting_list={}                    # To store posting list of each token in the corpus


for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()                                                # Converts docs to lower case
    tokens = tokenizer.tokenize(doc)                                 # tokenizing each and every document
    sw=stopwords.words('english')
    tokens = [stemmer.stem(word)
              for word in tokens
                   if word not in sw]                            # To remove stopwords and perform stemming, though PorterStemmer
    doc_feq= doc_feq + Counter(set(tokens))                                   # Particular token will be considered once by using 'counter set' and gets iterated when it comes in other 30 docs
    tf=Counter(tokens)
    tf_all[filename]=tf.copy()                                          # To create a copy for tf into tf_all for that particular file
    tf.clear()                                                       #  Clear tf, so that next document will be empty



def getidf(word):
    if doc_feq[word]==0:
        return -1
    return math.log(len(tf_all)/doc_feq[word],10)                # len(tf_all) to know number of docs,df[token] for each token's document frequency

def calculateWeight(filename, word):                                      # returns a weight of the token in a document without normalizing
    idf=getidf(word)
    return (1+log10(tf_all[filename][word]))*idf                       # tfs has term frequencies of docs in a multi-level dictionary

#Calculating tf-idf vectors and lengths_doc of the documents
for filename in tf_all:
    tfidfvector[filename]=Counter()                     # Creating vector to store tf-idf(weight) of each doc
    length=0
    for word in tf_all[filename]:
        weight = calculateWeight(filename, word)         # To calculate the weight of a token in a doc without normalization
        tfidfvector[filename][word]=weight             # Assign weight of a token in a file
        length = length + weight**2                         # calculate length for normalization
    lengths_doc[filename]=math.sqrt(length)

#Normalizing the weights
for filename in tfidfvector:
    for word in tfidfvector[filename]:
        tfidfvector[filename][word]/=  lengths_doc[filename]    # divides weight by the document's length
        if word not in posting_list:
            posting_list[word]=Counter()
        posting_list[word][filename]=tfidfvector[filename][word]                     # copying the normalized value into the posting list
        
        

def getweight(filename,word):
    return tfidfvector[filename][word]             # returns normalized weight of a token in a document

def query(querystring):                             # function returns the best match for a query
    query_tf={}
    query_length=0
    store_docs={}
    flag=0
    top_tenth={}
    cosine_simdocs=Counter()                          # initialize a counter for calculating cosine similarity of a token and a doc
    querystring=querystring.lower()                     # converting to lower case
    
    for word in querystring.split():
        word=stemmer.stem(word)               # stemm the token using PorterStemmer
        if word not in posting_list:          # if the token doesn't exist in vocabulary it will ignore  
            continue
        if getidf(word)==0:                    #if a token has idf = 0, all values in its postings list are zero. max 10 will be chosen randomly
            store_docs[word], weights = zip(*posting_list[word].most_common())         #to avoid that, we store all docs
        else:
            store_docs[word],weights = zip(*posting_list[word].most_common(10))        # taking top 10 in postings list, unpacks the sequence/collection and to pass the  item of this list to arguments

        top_tenth[word]=weights[9]                                                         # storing the upper bound of each token
        if flag==1:
            Docs_Common=set(store_docs[word]) & Docs_Common                                # Docs_Common keeps track of docs that have all tokens     
        else:
            Docs_Common=set(store_docs[word])
            flag=1
        query_tf[word]=1+log10(querystring.count(word))    # updating term freq of token in query
        query_length+=query_tf[word]**2                      # calculating length for normalizing the query tf later
    query_length=sqrt(query_length)
   

    for doc in tfidfvector:
        cosine_sim=0
        for word in query_tf:
            if doc in store_docs[word]:
                cosine_sim = cosine_sim + ((query_tf[word] / query_length) * posting_list[word][doc]  )     # calculate actual score if document is in top 10
            else:
                cosine_sim = cosine_sim + ((query_tf[word] / query_length) * top_tenth[word]    )                # otherwise, calculate its upper bound score
        cosine_simdocs[doc]=cosine_sim
    maximum=cosine_simdocs.most_common(1)                                                   # seeing which doc has the maximum value
    File_feedback,cossim_weight=zip(*maximum)

    try:
        if File_feedback[0] in Docs_Common:                                                 # if doc is present in Docs_Common and has actual score, return score
            return File_feedback[0],cossim_weight[0]
        else:
            return "fetch more",0.0                                                         # if upperbound score is greater, return fetch more
    except UnboundLocalError:                                                               # if none of the tokens are in vocabulary, return none
        return "None",0.0


# Printing the Sample Output for following getidf, getweight, query.

print("%.12f" % getidf("health"))
print("%.12f" % getidf("agenda"))
print("%.12f" % getidf("vector"))
print("%.12f" % getidf("reason"))
print("%.12f" % getidf("hispan"))
print("%.12f" % getidf("hispanic"))
print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
print("(%s, %.12f)" % query("health insurance wall street"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("vector entropy"))



"""
References:
https://www.w3schools.com/python/
www.stackoverflow.com
www.nltk.org
https://www.tutorialspoint.com/python/python_dictionary.htm
https://docs.python.org/3/library/collections.html#collections.Counter
"""









