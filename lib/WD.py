import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class topicscloud:
    def __init__(self,df):
        self.df = df
        self.text = ''
    
    def cloud_school(self,sch):
        df_sch = self.df[self.df.school == sch]['sentence_str']
        text = " ".join(review for review in df_sch)
        
        wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        res = pd.DataFrame.from_dict(wordcloud.words_,orient='index',columns=['times']).head(20).T
        
        return res
