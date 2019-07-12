# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:21:16 2019

@author: Big data
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:56:10 2019

@author: Big data
"""  
import numpy as np 
import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#t={"Runtime":"101min","budget_in_USD":1400000,"Production":"aaa","imdbVotes":0.5,"IMDBscore":0.6,"TomatoesScore":0.8,"Metascore":0.5,
#   "Theater_num":1000,"movie_2_before":50,"movie_1_before":50,"movie_0_before":50,"Actor_2_before":50,"Actor_1_before":50,
#   "Actor_0_before":50,"Genre":"Action","Language":"English","Country":"USA","classification":"R","Released":"27 JUN 2019"}

def model_yn(movielist):
    
    movielist['release_date_USA']=movielist['Released']
    

    want_keys={
            'classification', 'Runtime', 'budget_in_USD', 
           'release_date_USA', 'Genre', 'imdbVotes', 'IMDBscore', 'TomatoesScore',
           'Metascore', 'Theater_num', 'movie_2_before',
           'movie_1_before', 'movie_0_before', 'Actor_2_before',
           'Actor_1_before', 'Actor_0_before'}

    wanted_dict = {key : [val] for key ,val in movielist.items() if key in want_keys }   
    df = pd.DataFrame.from_dict(wanted_dict)

    for i in df.columns:
#        print(i)
        if df[i].isna().any():
            df[i][0] = str(0)
    
    df["runtime"] = df["Runtime"]
    df["runtime"] = df["runtime"].str.replace("min","")
    df["runtime"] = df["runtime"].astype("float")
    df = df.drop(["Runtime"],axis= 1)
    try:
        df["budget_in_USD"] = df["budget_in_USD"].str.replace('$',"").str.replace(",","")   
    except:
        pass
    df["budget_in_USD"] = df["budget_in_USD"].astype("float")
    
    try:
        df["imdbVotes"] = df["imdbVotes"][0].replace(",","")
    except:
        pass
    df["imdbVotes"] = df["imdbVotes"].astype("float")
    
    
    df['release_date_USA'] = df.release_date_USA.str.split(' ',expand=True)[1]
    
    #df["Cmovie_3_before"] = df["Cmovie_3_before"].astype("str").replace("error","0").astype("float")
    #df["Cmovie_2_before"] = df["Cmovie_2_before"].astype("str").replace("error","0").astype("float")
    #df["Cmovie_1_before"] = df["Cmovie_1_before"].astype("str").replace("error","0").astype("float")
    #df["Cmovie_0_before"] = df["Cmovie_0_before"].astype("str").replace("error","0").astype("float")
#    df["movie_3_before"] = df["movie_3_before"].astype("str").replace("error","0").astype("float")
    df["movie_2_before"] = df["movie_2_before"].astype("str").replace("error","0").astype("float")
    df["movie_1_before"] = df["movie_1_before"].astype("str").replace("error","0").astype("float")
    df["movie_0_before"] = df["movie_0_before"].astype("str").replace("error","0").astype("float")
#    df["Actor_3_before"] = df["Actor_3_before"].astype("str").replace("error","0").astype("float")
    df["Actor_2_before"] = df["Actor_2_before"].astype("str").replace("error","0").astype("float")
    df["Actor_1_before"] = df["Actor_1_before"].astype("str").replace("error","0").astype("float")
    df["Actor_0_before"] = df["Actor_0_before"].astype("str").replace("error","0").astype("float")
    df= df.join(pd.get_dummies(df["classification"]).astype("bool"))
    df = df.drop(["classification"], axis=1)
    
    df= df.join(pd.get_dummies(df["release_date_USA"]).astype("bool"))
    df = df.drop(["release_date_USA"], axis=1)
    
    
    #onhot encoding
    
    
    
    model_wanted_keys=[
            'runtime', 'budget_in_USD', 'imdbVotes', 'IMDBscore', 'TomatoesScore',
           'Metascore', 'Theater_num', 'movie_2_before',
           'movie_1_before', 'movie_0_before', 'Actor_2_before',
           'Actor_1_before', 'Actor_0_before', 'G', 'NC-17', 'NotRated', 'PG',
           'PG-13', 'R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'TV-Y', 'TV-Y7',
           'TV-Y7-FV', 'Unrated', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun',
           'Mar', 'May', 'Nov', 'Oct', 'Sep', 'Action', 'Adventure', 'Animation',
           'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
           'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
           'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
    data = [0]*len(model_wanted_keys)
    
    df_model = pd.DataFrame(data=[data],columns= model_wanted_keys)
    df_model.update(df)
    
    with open('RandomForest.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        #测试读取后的Model
    return clf2.predict(df_model)[0]


def predict_loss(m):
    import pymongo
    import pandas as pd
    import numpy as np
    import re
    import xgboost as xgb

    
    wanted_keys={"Runtime","budget_in_USD","Production","imdbVotes","IMDBscore","TomatoesScore","Metascore","Theater_num",
                 "movie_2_before","movie_1_before","movie_0_before","Actor_2_before","Actor_1_before","Actor_0_before",
                 "Genre","Language","Country","classification"}
    
    wanted_dict  = {key : [val] for key ,val in m.items() if key in wanted_keys}
    df_company = pd.read_csv("company_detail.csv")
    company_dict = {}
    for i, s in df_company.iterrows():
        company_dict[s["company"]] = s["avg"]
    key = list(company_dict.keys())
    
    regex_pat = re.compile(r"[^a-zA-Z0-9]+", flags=re.IGNORECASE)
    wanted_dict["Production"][0] = re.sub(regex_pat,'', wanted_dict["Production"][0]).lower()
    wanted_dict["Production"] = wanted_dict["Production"][0]
    for x in key:
        if wanted_dict["Production"].find(x) != -1:
           wanted_dict["Production"] = company_dict[wanted_dict["Production"]] 
           break 
   # print("type",type(wanted_dict["Production"]),type(wanted_dict["Production"]) == 'str',wanted_dict["Production"])
    if type(wanted_dict["Production"]) == str:
        wanted_dict["Production"]= 200000
    
    df = pd.DataFrame.from_dict(wanted_dict)        
    cleaned = df.Genre.str.split(',', expand=True).stack()
    cleaned = cleaned.apply(lambda x:x.replace(" ",""))
    genre_enc = pd.get_dummies(cleaned, prefix='g').groupby(level=0,sort=False).sum()
    
    cleaned = df.Language.str.split(',', expand=True).stack()
    cleaned = cleaned.apply(lambda x:x.replace(" ",""))
    #drop_lang = (cleaned.value_counts()[cleaned.value_counts()<=5]).index.tolist()
    #for l in range(len(cleaned)):
    #    if cleaned.iloc[l] in drop_lang:
    #        cleaned.iloc[l] = "Other_language"
    language_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()
    
    cleaned = df.Country.str.split(',', expand=True).stack()
    cleaned = cleaned.apply(lambda x:x.replace(" ",""))
    country_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()   
        
    df["Runtime"] = df["Runtime"].str.replace("min","")
    try:
        df["imdbVotes"]=df["imdbVotes"].str.replace(",","")
    except:
        pass
    for e in df.columns:
        if "movie" or "Actor" in e:
            df[e] = df[e].replace("error","0")
    cla_enc = pd.get_dummies(df["classification"])
    df = df.drop(["classification","Genre","Language","Country"],axis=1)
    
    for i in df.columns:
        try:
            df[i] =pd.to_numeric(df[i], downcast='float')
            #df["budget_in_USD"] = np.log(df["budget_in_USD"])
        except:
            pass
                
    
    #df["Production"] = np.log(df["Production"])   
    df["Theater_num"] = np.log(df["Theater_num"]) 
    df = pd.concat([df.reset_index(drop=True),genre_enc.reset_index(drop=True),
                    country_enc.reset_index(drop=True),language_enc.reset_index(drop=True),
                    cla_enc.reset_index(drop=True)
                    ],
                    axis=1) 
    
    
    
    feature = ['budget_in_USD','Production','imdbVotes','IMDBscore','TomatoesScore','Metascore','Theater_num',
 'movie_2_before','movie_1_before','movie_0_before',
 'Actor_2_before','Actor_1_before','Actor_0_before','Runtime','g_Action','g_Adventure','g_Animation','g_Biography',
 'g_Comedy','g_Crime','g_Drama','g_Family','g_Fantasy','g_History',
 'g_Horror','g_Music','g_Musical','g_Mystery','g_Romance','g_Sci-Fi',
 'g_Sport','g_Thriller','g_War','g_Western','L_Algeria','L_Angola','L_Argentina','L_Australia',
 'L_Austria','L_Belgium','L_BosniaandHerzegovina','L_Botswana','L_Brazil','L_BritishVirginIslands','L_Bulgaria','L_Cambodia',
 'L_Canada','L_CaymanIslands','L_Chile','L_China','L_Colombia',
 'L_Croatia','L_Cyprus','L_CzechRepublic','L_Denmark','L_Egypt','L_Estonia','L_Finland','L_France','L_Georgia',
 'L_Germany','L_Greece','L_HongKong','L_Hungary','L_Iceland','L_India','L_Indonesia','L_Iran',
 'L_Ireland','L_IsleOfMan','L_Israel','L_Italy','L_Japan','L_Jordan','L_Kazakhstan','L_Kenya','L_Lebanon','L_Liechtenstein',
 'L_Lithuania','L_Luxembourg','L_Malta','L_Mexico','L_Monaco','L_Mongolia','L_Morocco',
 'L_Myanmar','L_Nepal','L_Netherlands','L_NewZealand','L_Nigeria','L_Norway','L_Palestine','L_Panama','L_Paraguay','L_Peru',
 'L_Philippines','L_Poland','L_Portugal','L_PuertoRico','L_Qatar','L_Romania','L_Russia','L_SaudiArabia','L_Serbia','L_Singapore','L_Slovakia',
 'L_Slovenia','L_SouthAfrica','L_SouthKorea','L_Spain','L_Sweden','L_Switzerland','L_Taiwan','L_Thailand',
 'L_Tunisia','L_Turkey','L_UK','L_USA','L_Ukraine','L_UnitedArabEmirates','L_Uruguay','L_Venezuela','L_Vietnam','G','NC-17','NotRated','PG',
 'PG-13','R','TV-14','TV-MA','TV-PG','Unrated']
    
    #feature2 = ['IMDBscore','TV-MA','g_Action','L_Canada','L_HongKong','L_Mexico','L_Jordan','L_India','movie_2_before','L_Chile','L_Peru',
    # 'g_Crime','g_Comedy','L_Denmark','L_Venezuela','L_Mongolia','L_Portugal','Actor_1_before','g_Music','L_Iran','L_Cyprus','NotRated',
    # 'L_BritishVirginIslands','L_CaymanIslands','L_Iceland','L_Kazakhstan','L_Romania','L_Palestine','TomatoesScore','L_Philippines','L_SaudiArabia',
    # 'L_Argentina','budget_in_USD','L_Italy','L_Netherlands','L_Spain','L_UK','g_Drama','L_Malta','Unrated','L_Singapore','L_France',
    # 'L_Austria','L_Egypt','L_Kenya','runtime','g_Romance','L_Switzerland','L_SouthAfrica','imdbVotes','L_Lithuania','g_Sport',
    # 'movie_0_before','TV-14','L_Slovenia','movie_1_before','L_Nigeria','L_PuertoRico','Actor_2_before','L_Japan','L_Indonesia','g_Fantasy',
    # 'L_Israel','NC-17','L_Bulgaria','L_Luxembourg','L_Estonia','L_Algeria','g_Musical','L_IsleOfMan','L_Poland','L_Ireland','Actor_0_before',
    # 'g_War','L_Paraguay','L_Cambodia','Metascore','L_Monaco','L_Angola','L_Turkey','L_Georgia','L_Serbia','L_Panama','Production','L_Australia',
    # 'L_Colombia','g_Adventure','PG','L_Uruguay','PG-13','g_History','L_Morocco','L_Hungary','L_Myanmar','L_Ukraine','G','TV-PG','g_Horror',
    # 'L_Brazil','R','L_Vietnam','L_Lebanon','L_Belgium','g_Sci-Fi','L_Botswana','g_Western','L_Tunisia','L_BosniaandHerzegovina','L_Qatar',
    # 'g_Thriller','L_Finland','L_Russia','g_Biography','L_SouthKorea','L_Germany','L_Liechtenstein','L_NewZealand','L_Thailand','g_Animation',
    # 'L_Croatia','L_Greece','L_Sweden','L_UnitedArabEmirates','L_CzechRepublic','L_USA','g_Mystery','L_Taiwan','L_Nepal','L_Slovakia',
    # 'L_Norway','g_Family','Theater_num','L_China']
    
    data = [0]*len(feature)
    df_model = pd.DataFrame(data=[data],columns=feature)
    df_model.update(df)
#    feature_list = ["f"+str(i) for i in range(len(feature))]
#    df_model.columns = feature_list
    #for f in feature2:
    #    if f not in df.columns.tolist():
    #        df_append = pd.DataFrame.from_dict({f:[0]})
    #        df=pd.concat([df,df_append],axis=1)
    #import joblib
    #xgb = joblib.load("modeltest")
    import pickle
    loaded_model = pickle.load(open("loss.pickle.dat", "rb"))
    y_pred = loaded_model.predict(df_model)
    y_pred = np.exp(y_pred)
    return y_pred

def predict_gain(m):
    import pymongo
    import pandas as pd
    import numpy as np
    import re
    import xgboost as xgb

    
    wanted_keys={"Runtime","budget_in_USD","Production","imdbVotes","IMDBscore","TomatoesScore","Metascore","Theater_num",
                 "movie_2_before","movie_1_before","movie_0_before","Actor_2_before","Actor_1_before","Actor_0_before",
                 "Genre","Language","Country","classification"}
    
    wanted_dict  = {key : [val] for key ,val in m.items() if key in wanted_keys}
    df_company = pd.read_csv("company_detail.csv")
    company_dict = {}
    for i, s in df_company.iterrows():
        company_dict[s["company"]] = s["avg"]
    key = list(company_dict.keys())
    
    regex_pat = re.compile(r"[^a-zA-Z0-9]+", flags=re.IGNORECASE)
    wanted_dict["Production"][0] = re.sub(regex_pat,'', wanted_dict["Production"][0]).lower()
    wanted_dict["Production"] = wanted_dict["Production"][0]
    for x in key:
        if wanted_dict["Production"].find(x) != -1:
           wanted_dict["Production"] = company_dict[wanted_dict["Production"]] 
           break 
    #print("type",type(wanted_dict["Production"]),type(wanted_dict["Production"]) == 'str',wanted_dict["Production"])
    if type(wanted_dict["Production"]) == str:
        wanted_dict["Production"]= 200000
    
    df = pd.DataFrame.from_dict(wanted_dict)        
    cleaned = df.Genre.str.split(',', expand=True).stack()
    cleaned = cleaned.apply(lambda x:x.replace(" ",""))
    genre_enc = pd.get_dummies(cleaned, prefix='g').groupby(level=0,sort=False).sum()
    
    cleaned = df.Language.str.split(',', expand=True).stack()
    cleaned = cleaned.apply(lambda x:x.replace(" ",""))
    #drop_lang = (cleaned.value_counts()[cleaned.value_counts()<=5]).index.tolist()
    #for l in range(len(cleaned)):
    #    if cleaned.iloc[l] in drop_lang:
    #        cleaned.iloc[l] = "Other_language"
    language_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()
    
    cleaned = df.Country.str.split(',', expand=True).stack()
    cleaned = cleaned.apply(lambda x:x.replace(" ",""))
    country_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()   
        
    df["Runtime"] = df["Runtime"].str.replace("min","")
    try:
        df["imdbVotes"]=df["imdbVotes"].str.replace(",","")
    except:
        pass
    for e in df.columns:
        if "movie" or "Actor" in e:
            df[e] = df[e].replace("error","0")
    cla_enc = pd.get_dummies(df["classification"])
    df = df.drop(["classification","Genre","Language","Country"],axis=1)
    
    for i in df.columns:

        try:
            df[i] =pd.to_numeric(df[i], downcast='float')
            #df["budget_in_USD"] = np.log(df["budget_in_USD"])
        except:
            pass
        
    
    #df["Production"] = np.log(df["Production"])   
    df["Theater_num"] = np.log(df["Theater_num"]) 
    df = pd.concat([df.reset_index(drop=True),genre_enc.reset_index(drop=True),
                    country_enc.reset_index(drop=True),language_enc.reset_index(drop=True),
                    cla_enc.reset_index(drop=True)
                    ],
                    axis=1) 
    
    
    
    feature = ['budget_in_USD', 'Production', 'imdbVotes', 'IMDBscore',
       'TomatoesScore', 'Metascore', 'Theater_num', 'movie_2_before',
       'movie_1_before', 'movie_0_before', 'Actor_2_before', 'Actor_1_before',
       'Actor_0_before', 'Runtime', 'g_Action', 'g_Adventure', 'g_Animation',
       'g_Biography', 'g_Comedy', 'g_Crime', 'g_Drama', 'g_Family',
       'g_Fantasy', 'g_History', 'g_Horror', 'g_Music', 'g_Musical',
       'g_Mystery', 'g_Romance', 'g_Sci-Fi', 'g_Sport', 'g_Thriller', 'g_War',
       'g_Western', 'L_Argentina', 'L_Australia', 'L_Austria', 'L_Bahamas',
       'L_Belgium', 'L_Brazil', 'L_Bulgaria', 'L_Cambodia', 'L_Canada',
       'L_Chile', 'L_China', 'L_Colombia', 'L_CzechRepublic', 'L_Denmark',
       'L_DominicanRepublic', 'L_Finland', 'L_France', 'L_Germany', 'L_Greece',
       'L_HongKong', 'L_Hungary', 'L_Iceland', 'L_India', 'L_Indonesia',
       'L_Iran', 'L_Ireland', 'L_IsleOfMan', 'L_Israel', 'L_Italy', 'L_Japan',
       'L_Kenya', 'L_Luxembourg', 'L_Malaysia', 'L_Malta', 'L_Mexico',
       'L_Morocco', 'L_Netherlands', 'L_NewZealand', 'L_Norway',
       'L_Philippines', 'L_Poland', 'L_Romania', 'L_Russia', 'L_Serbia',
       'L_Singapore', 'L_Slovakia', 'L_SouthAfrica', 'L_SouthKorea', 'L_Spain',
       'L_Sweden', 'L_Switzerland', 'L_Taiwan', 'L_Thailand', 'L_Turkey',
       'L_UK', 'L_USA', 'L_Ukraine', 'L_UnitedArabEmirates', 'G', 'NotRated',
       'PG', 'PG-13', 'R', 'Unrated']
    
    #feature2 = ['IMDBscore','TV-MA','g_Action','L_Canada','L_HongKong','L_Mexico','L_Jordan','L_India','movie_2_before','L_Chile','L_Peru',
    # 'g_Crime','g_Comedy','L_Denmark','L_Venezuela','L_Mongolia','L_Portugal','Actor_1_before','g_Music','L_Iran','L_Cyprus','NotRated',
    # 'L_BritishVirginIslands','L_CaymanIslands','L_Iceland','L_Kazakhstan','L_Romania','L_Palestine','TomatoesScore','L_Philippines','L_SaudiArabia',
    # 'L_Argentina','budget_in_USD','L_Italy','L_Netherlands','L_Spain','L_UK','g_Drama','L_Malta','Unrated','L_Singapore','L_France',
    # 'L_Austria','L_Egypt','L_Kenya','runtime','g_Romance','L_Switzerland','L_SouthAfrica','imdbVotes','L_Lithuania','g_Sport',
    # 'movie_0_before','TV-14','L_Slovenia','movie_1_before','L_Nigeria','L_PuertoRico','Actor_2_before','L_Japan','L_Indonesia','g_Fantasy',
    # 'L_Israel','NC-17','L_Bulgaria','L_Luxembourg','L_Estonia','L_Algeria','g_Musical','L_IsleOfMan','L_Poland','L_Ireland','Actor_0_before',
    # 'g_War','L_Paraguay','L_Cambodia','Metascore','L_Monaco','L_Angola','L_Turkey','L_Georgia','L_Serbia','L_Panama','Production','L_Australia',
    # 'L_Colombia','g_Adventure','PG','L_Uruguay','PG-13','g_History','L_Morocco','L_Hungary','L_Myanmar','L_Ukraine','G','TV-PG','g_Horror',
    # 'L_Brazil','R','L_Vietnam','L_Lebanon','L_Belgium','g_Sci-Fi','L_Botswana','g_Western','L_Tunisia','L_BosniaandHerzegovina','L_Qatar',
    # 'g_Thriller','L_Finland','L_Russia','g_Biography','L_SouthKorea','L_Germany','L_Liechtenstein','L_NewZealand','L_Thailand','g_Animation',
    # 'L_Croatia','L_Greece','L_Sweden','L_UnitedArabEmirates','L_CzechRepublic','L_USA','g_Mystery','L_Taiwan','L_Nepal','L_Slovakia',
    # 'L_Norway','g_Family','Theater_num','L_China']
    
    data = [0]*len(feature)
    df_model = pd.DataFrame(data=[data],columns=feature)
    df_model.update(df)
#    feature_list = ["f"+str(i) for i in range(len(feature))]
#    df_model.columns = feature_list
    #for f in feature2:
    #    if f not in df.columns.tolist():
    #        df_append = pd.DataFrame.from_dict({f:[0]})
    #        df=pd.concat([df,df_append],axis=1)
    #import joblib
    #xgb = joblib.load("modeltest")
    import pickle
    loaded_model = pickle.load(open("gain.pickle.dat", "rb"))
    y_pred = loaded_model.predict(df_model)
    y_pred = np.exp(y_pred)
    return y_pred

def run_model(t):
    ans = model_yn(t)
    if ans:
       return("您會賺錢",predict_gain(t))
    else:
       return("您會虧錢",predict_loss(t))
if __name__ == "__main__":
    a,b=run_model(t)
    




    
    
#    t={}
#    print(model_yn(t))
#    if model_yn(t) == 0:
#        print(predict_loss(t))
#    else:
#        print(predict_gain(t))














