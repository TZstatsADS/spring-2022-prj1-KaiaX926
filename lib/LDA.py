import warnings
warnings.simplefilter("ignore")

from glob import glob
import re
import string
import funcy as fp
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
import nltk
import pandas as pd
from gensim.utils import simple_preprocess
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

import numpy as np
import pandas as pd

class LDA_phil:
    def __init__(self,df,sch):
        self.df = df[df.school == sch]
        self.name = sch
    
    
    def process(self):
        pre_df = self.df[['school','sentence_str','tokenized_txt']].set_index('school')
        pre_df['gensim_tokenized'] = pre_df['sentence_str'].map(lambda x: simple_preprocess(x.lower(),deacc=True,
                                                        max_len=100))
        dictionary, corpus = self.prep_corpus(pre_df['gensim_tokenized'])
        #filename = "%s.csv" % name
        MmCorpus.serialize('../output/%s.mm'% self.name, corpus)
        dictionary.save('../output/%s.dict'% self.name)
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)#, passes=10)
                                      
        lda.save('../output/%s.model'% self.name)
        vis_data = gensimvis.prepare(lda, corpus, dictionary)
        
        return vis_data
        
    
    def nltk_stopwords(self):
        return set(nltk.corpus.stopwords.words('english'))

    def prep_corpus(self,docs, additional_stopwords=set(), no_below=5, no_above=0.5):
        print('Building dictionary...')
        dictionary = Dictionary(docs)
        stopwords = self.nltk_stopwords().union(additional_stopwords)
        stopword_ids = map(dictionary.token2id.get, stopwords)
        dictionary.filter_tokens(stopword_ids)
        dictionary.compactify()
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
        dictionary.compactify()

        print('Building corpus...')
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        return dictionary, corpus
    
