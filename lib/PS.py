from flair.models import TextClassifier
from flair.data import Sentence
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

class philosophy_sentiment: 
    def __init__(self,df):
        self.df = df
        self.school_list = list(self.df.school.drop_duplicates().reset_index(drop = 'True'))
        self.df_auth = self.df[['school','author']].drop_duplicates().reset_index(drop = 'True')
        self.classifier = TextClassifier.load('en-sentiment')

    def matrix(self):
        attitude_matrix,times_matrix = [],[]
        
        for sch in self.school_list:
            print(sch)
            attitude_matrix.append(self.cited_author(sch)[0])
            times_matrix.append(self.cited_author(sch)[1])
        
        attitude_matrix = pd.DataFrame(data = attitude_matrix,columns = self.school_list, index = self.school_list)
        attitude_matrix = attitude_matrix.add_suffix('_act')
        
        times_matrix = pd.DataFrame(data = times_matrix,columns = self.school_list, index = self.school_list)
        times_matrix = times_matrix.add_suffix('_act')
        
        return attitude_matrix,times_matrix
 
    def cited_author(self,school):
        author = list(self.df_auth[self.df_auth.school == school].author)
        author.append(school)
        res = [[0,0] for i in range(len(self.school_list))]
        for i in range(len(self.df)):
            if any(ele in self.df.sentence_str.loc[i] for ele in author):
                sentence = Sentence(self.df.sentence_str.loc[i])
                self.classifier.predict(sentence)
                
                s = self.df.school.loc[i]
                index = self.school_list.index(s)
                if sentence.labels[0].value == 'NEGATIVE':
                    score = - sentence.labels[0].score
                else:
                    score = sentence.labels[0].score
                res[index][0] += 1
                res[index][1] += score
 
        ans,times = [],[]
        for item in res:
            if item[0] != 0:
                ans.append(item[1]/item[0])
            else:
                ans.append(0)
            times.append(item[0])

        return ans,times
