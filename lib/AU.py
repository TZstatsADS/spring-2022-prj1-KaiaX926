import warnings
warnings.simplefilter("ignore")
from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np
import pandas as pd

class author_analysis:
    def __init__(self,df):
        self.df = df
        self.school_list = list(df.school.drop_duplicates().reset_index(drop = 'True'))
        self.df_auth = df[['school','author']].drop_duplicates().reset_index(drop = 'True')
        self.classifier = TextClassifier.load('en-sentiment')
    
    def popular_author(self,sch_act,sch):
        au_df = self.df[self.df['school'] == sch_act].reset_index(drop = 'True')
        au_sch = self.df_auth[self.df_auth.school == sch].reset_index(drop = 'True')
        au_sch_list = list(au_sch['author'])
        au_sch['Citation_times'] = 0
        for i in range(len(au_df)):
            for au in au_sch_list:
                if au in au_df.sentence_str[i]:
                    au_sch.Citation_times[au_sch['author'] == au] += 1
                    #print(au_sch,au_df.sentence_str[i])
            
        return au_sch.sort_values('Citation_times',ascending = False)
    
    
    def the_author(self,author):
        times = [0 for i in range(len(self.school_list))]
        attitude = [0 for i in range(len(self.school_list))]
        for i in range(len(self.df)):
            if author in self.df.sentence_str.loc[i]:
                sentence = Sentence(self.df.sentence_str.loc[i])
                self.classifier.predict(sentence)
                
                s = self.df.school.loc[i]
                index = self.school_list.index(s)
                if sentence.labels[0].value == 'NEGATIVE':
                    score = - sentence.labels[0].score
                else:
                    score = sentence.labels[0].score
                times[index] += 1
                attitude[index] += score
 
        for i in range(len(times)):
            if times[i] != 0:
                attitude[i] = attitude[i]/times[i]
        
        res = pd.DataFrame(columns = self.school_list,index = ['Citation_times','attitude'])
        res.loc['Citation_times'] = times
        res.loc['attitude'] = attitude
        
        return res
