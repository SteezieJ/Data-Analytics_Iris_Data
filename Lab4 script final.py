import pandas as pd
import numpy as np
import warnings
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


warnings.filterwarnings('ignore')

# --------------------- Q 1 ---------------------

def GetData():
    Rev = pd.read_excel('Iris_Data.xlsx',header=None,names=['sepal_length','sepal_width','petal_length','petal_width','Species'],sheet_name="Sheet1")
    Rev['Species'] = Rev.Species.apply(lambda x: x[1:-1].capitalize())
    return Rev
print('-- Fisher Iris Dataset --')
df = GetData()
print(df.head())
print('\n')
print(' -----          Q1          -----')
def GetAvg_Var(dataframe,feature,feature_class):  # Function Helps to Find Mean and Variance
    variance = np.var(dataframe[feature])
    mean = np.mean(dataframe[feature])
    all_cls = dataframe[feature_class].unique()
    print(' -------- For Population --------')
    print(f"Mean for {feature} is {round(mean,2)}")
    print(f"Variance for {feature} is {round(variance,2)}")
    print('\n')
    print(' -------- For Classes --------')
    for i in all_cls:
        cls = dataframe[dataframe[feature_class] == i]
        variance_cls = np.var(cls[feature])
        mean_cls = np.mean(cls[feature])
        print(f"Mean for {feature} is {round(mean_cls,2)}, for {feature_class} is {i} ")
        print(f"Variance for {feature} is {round(variance_cls,2)}, for {feature_class} is {i} ")
        print('\n')
    
for i in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    GetAvg_Var(df,i,"Species")
    print('-------------------------------------------------')
    print('\n')

# --------------------- Q 2 ---------------------
    print(' -----          Q2          -----')
def GenerateNoisyDataset(df):
    SL = []
    SW = []
    PL = []
    PW = []
    for i in df.Species.unique():
        mean = 0
        sepal_len_std,sepal_wid_std,petal_len_std,petal_wid_std = df[df.Species == i].std()
        sepal_length = list(np.random.normal(0, sepal_len_std, 50)+df[df.Species == i].sepal_length)
        sepal_width = list(np.random.normal(0, sepal_wid_std, 50)+df[df.Species == i].sepal_width)
        petal_length = list(np.random.normal(0, petal_len_std, 50)+df[df.Species == i].petal_length)
        petal_width = list(np.random.normal(0, petal_wid_std, 50)+df[df.Species == i].petal_width)
        SL.append(sepal_length)
        SW.append(sepal_width)
        PL.append(petal_length)
        PW.append(petal_width)
        
    SeLe = pd.DataFrame(np.transpose(SL),columns=[df.Species.unique()])
    SeWi = pd.DataFrame(np.transpose(SW),columns=[df.Species.unique()])
    PeLe = pd.DataFrame(np.transpose(PL),columns=[df.Species.unique()])
    PeWi = pd.DataFrame(np.transpose(PW),columns=[df.Species.unique()])
    
    Sepal_Lenght_S = []
    Sepal_Lenght_Ve = []
    Sepal_Lenght_Vi = []
    Sepal_width_S = []
    Sepal_width_Ve = []
    Sepal_width_Vi = []
    petal_Lenght_S = []
    petal_Lenght_Ve = []
    petal_Lenght_Vi = []
    petal_width_S = []
    petal_width_Ve = []
    petal_width_Vi = []
    
    for i in range(SeLe.shape[0]):
        var = SeLe.iloc[i]
        Setosa = var.values[0]
        Versicolor = var.values[1]
        Virginica = var.values[2]
        Sepal_Lenght_S.append(Setosa)
        Sepal_Lenght_Ve.append(Versicolor)
        Sepal_Lenght_Vi.append(Virginica)
    for i in range(SeWi.shape[0]):
        var = SeWi.iloc[i]
        Setosa = var.values[0]
        Versicolor = var.values[1]
        Virginica = var.values[2]
        Sepal_width_S.append(Setosa)
        Sepal_width_Ve.append(Versicolor)
        Sepal_width_Vi.append(Virginica)
    for i in range(PeLe.shape[0]):
        var = PeLe.iloc[i]
        Setosa = var.values[0]
        Versicolor = var.values[1]
        Virginica = var.values[2]
        petal_Lenght_S.append(Setosa)
        petal_Lenght_Ve.append(Versicolor)
        petal_Lenght_Vi.append(Virginica)
    for i in range(PeWi.shape[0]):
        var = PeWi.iloc[i]
        Setosa = var.values[0]
        Versicolor = var.values[1]
        Virginica = var.values[2]
        petal_width_S.append(Setosa)
        petal_width_Ve.append(Versicolor)
        petal_width_Vi.append(Virginica)
    dfl1 = pd.DataFrame(Sepal_Lenght_S,columns=['Sepal Length'])
    dfl1['Sepal width'] = Sepal_width_S
    dfl1['Petal Length'] = petal_Lenght_S
    dfl1['Petal width'] = petal_width_S
    dfl1['Species'] = ["Setosa" for i in range(len(dfl1))]

    dfl2 = pd.DataFrame(Sepal_Lenght_Ve,columns=['Sepal Length'])
    dfl2['Sepal width'] = Sepal_width_Ve
    dfl2['Petal Length'] = petal_Lenght_Ve
    dfl2['Petal width'] = petal_width_Ve
    dfl2['Species'] = ["Versicolor" for i in range(len(dfl1))]

    dfl3 = pd.DataFrame(Sepal_Lenght_Vi,columns=['Sepal Length'])
    dfl3['Sepal width'] = Sepal_width_Vi
    dfl3['Petal Length'] = petal_Lenght_Vi
    dfl3['Petal width'] = petal_width_Vi
    dfl3['Species'] = ["Virginica" for i in range(len(dfl1))]

    df2 = pd.concat([dfl1,dfl2,dfl3],axis=0).reset_index(drop=True)
    return df2

df2= GenerateNoisyDataset(df)

print('------ Fisher Iris Noisy Dataset ------ ')
print(df2.head())
print('\n')


for i in ['Sepal Length', 'Sepal width', 'Petal Length', 'Petal width']:
    GetAvg_Var(df2,i,"Species")
    print('-------------------------------------------------')
    print('\n')

def GetPopStats():   # Get Population Statistics Difference
    st1= df.describe()
    st2 = df2.describe()
    
    Pop_stats = pd.DataFrame(st1['sepal_length']-st2['Sepal Length'],columns=['sepal_length'])
    Pop_stats['Sepal width'] = st1['sepal_width']-st2['Sepal width']
    Pop_stats['Petal Length'] = st1['petal_length']-st2['Petal Length']
    Pop_stats['Petal width'] = st1['petal_width']-st2['Petal width']
    
    return Pop_stats

def GetClsStats(): # Get Classes Statistics Difference
    for i in df.Species.unique():
        st1= df[df.Species == i].describe()
        st2 = df2[df2.Species == i].describe()
        print(f" ------------- {i} -------------")
        Cls_stat = pd.DataFrame(st1['sepal_length']-st2['Sepal Length'],columns=['sepal_length'])
        Cls_stat['Sepal width'] = st1['sepal_width']-st2['Sepal width']
        Cls_stat['Petal Length'] = st1['petal_length']-st2['Petal Length']
        Cls_stat['Petal width'] = st1['petal_width']-st2['Petal width']
        print(Cls_stat)
        print("--------------------------------------------------------")

print(' --------- Population Statistical Difference --------- ')
print(GetPopStats())
print('--------------------------------------------------------')
print('\n')
print(' --------- Classes Statistical Difference --------- ')
GetClsStats()

# ---------------------------------- Q 3 -------------------------------------------
print('\n -----          Q3          -----\n')
def GetT_Statistic(dataframe,class_var,List_var):
    values = dataframe[class_var].unique()
    Pop_mean = dataframe.mean().values
    Spe = []
    T_sts = []
    Cols = []
    P_val = []
    
    for i in values:
        for j in List_var:
            Ls = dataframe[dataframe[class_var]==i][j]
            Pop_mean = dataframe[j].mean()
            T_Statistics,p_value = stats.ttest_1samp(Ls,Pop_mean)
            T_sts.append(T_Statistics)
            Cols.append(j)
            Spe.append(i)
            P_val.append(p_value)
    Ret = pd.DataFrame(Spe,columns=['Species'])
    Ret['Follower Parameter'] = Cols
    Ret['T_Statistics'] = T_sts
    Ret['P_value'] = P_val
    return Ret
def Get_Diff(val1,val2,cols,cols2):
    diff = val1[cols] - val2[cols]
    diff2 = pd.concat([val1[cols2],diff],axis=1)
    return diff2

print(' ------------ T_Statistics of Original Dataset ------------ ')
Original_Tvalues = GetT_Statistic(df,"Species",['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print(Original_Tvalues)
print("---------------------------------------------------------")
print("\n")
print(' ------------ T_Statistics of Original Dataset ------------ ')
Noisy_data_Tvalues = GetT_Statistic(df2,"Species",['Sepal Length', 'Sepal width', 'Petal Length', 'Petal width'])
print(Noisy_data_Tvalues)
print("---------------------------------------------------------")
print('\n')
print(" ---------------- T_statistics and P_value Difference from Original to Noisy Data ---------------- ")
print(Get_Diff(Original_Tvalues,Noisy_data_Tvalues,['T_Statistics','P_value'], ['Species', 'Follower Parameter']))
print("---------------------------------------------------------")
print('\n')

# --------------------------- Q 4 -----------------------------------------
print(' -----          Q4          -----')
def Get_Anova_test(G1,G2,G3):
    Degree_of_Freedom_bet = 3-1
    Degree_of_Freedom_within = len(G1)+len(G2)+len(G3)-3
    F_Critical = stats.f.ppf(1-0.05,dfn=Degree_of_Freedom_bet, dfd=Degree_of_Freedom_within)
    F_statistic,p_value = stats.f_oneway(G1,G2,G3)
    return F_statistic,F_Critical,p_value
def Get_Anova_table(Dataframe,Tab):
    Anova_Table = pd.DataFrame(['Sepal Length', 'Sepal width', 'Petal Length', 'Petal width'],columns=['Parameter'])
    F = []
    F_c = []
    p_v = []
    
    for i in Tab:
        F_statistic,F_Critical,p_value = Get_Anova_test(Dataframe[Dataframe.Species == 'Setosa'][i],Dataframe[Dataframe.Species == 'Versicolor'][i],Dataframe[Dataframe.Species == 'Virginica'][i])
        F.append(F_statistic)
        F_c.append(F_Critical)
        p_v.append(p_value)

    Anova_Table['F_Statistic']  = F
    Anova_Table['F Critical']  = F_c    
    Anova_Table['P value']  = p_v
    Anova_Table['Alpha Value']  = [0.05 for i in range(Anova_Table.shape[0])]
    Anova_Table['F_Stats > F_Critical'] = np.where(Anova_Table['F_Statistic']>Anova_Table['F Critical'],True,False)
    Anova_Table['P_value < Alpha'] = np.where(Anova_Table['Alpha Value']>Anova_Table['P value'],True,False)
    Anova_Table['Reject or Accept Null Hypothesis'] = np.where(Anova_Table['P_value < Alpha'] == True,'Reject Null Hypothesis','Accept Null Hypothesis')
    return Anova_Table
def GenerateScatter(var1,var2,title):
    plt.figure(figsize=(12,7))
    sns.scatterplot(var1,var2)
    plt.title(title)
    return plt.show()

print('\n')
print('Null Hypothesis: All Means of Three Group (i.e Species) are Identical, Which Means that All Species have Same Sepal, Petal, length and width')
print('Alternative Hypothesis: At Least Mean of One Group (i.e Species) have Different Mean, which Means that Species have difference in Sepal, Petal, length and Width')
print('\n')
print(' ----------- ANOVA Results for Original Table ----------- ')

Anv_Tab_Original = Get_Anova_table(df, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print(Anv_Tab_Original)
print(" ----------------------------------------------- ")
print('\n')
Top2Or = list(Anv_Tab_Original.sort_values(by="F_Statistic",ascending=False).head(2).Parameter.values)
Least2Or = list(Anv_Tab_Original.sort_values(by="F_Statistic",ascending=True).head(2).Parameter.values)
print(f'Top 2 Prominent Features are : {Top2Or}')
print(f'Least 2 Prominent Features are : {Least2Or}')
print('\n')
df3 = df.copy()
df3['Sepal Length'] = df3.sepal_length
df3['Sepal width'] = df3.sepal_width
df3['Petal Length'] = df3.petal_length
df3['Petal width'] = df3.petal_width

GenerateScatter(df3[Top2Or[0]],df3[Top2Or[1]],"Top 2 Prominenet Features in Original Dataset")
GenerateScatter(df3[Least2Or[0]],df3[Least2Or[1]],"Least 2 Prominenet Features in Original Dataset")

print(' ----------- ANOVA Results for Noisy Table ----------- ')
Anv_Tab_Noisy = Get_Anova_table(df2, ['Sepal Length', 'Sepal width', 'Petal Length', 'Petal width',])
print(Anv_Tab_Noisy)
print(" ----------------------------------------------- ")
print('\n')
Top2No = list(Anv_Tab_Noisy.sort_values(by="F_Statistic",ascending=False).head(2).Parameter.values)
Least2No = list(Anv_Tab_Noisy.sort_values(by="F_Statistic",ascending=True).head(2).Parameter.values)
print(f'Top 2 Prominent Features are : {Top2No}')
print(f'Least 2 Prominent Features are : {Least2No}')
print('\n')
GenerateScatter(df2[Top2No[0]],df2[Top2No[1]],"Top 2 Prominent Features in Noisy Dataset")
GenerateScatter(df2[Least2No[0]],df2[Least2No[1]],"Least 2 Prominent Features in Noisy Dataset")
print('\n')
print('Explanation: We have Reject Null Hypothesis, and Accepting Alternative Hypothesis, It is Proven by Both Value F_Statistics and P_value')
print('\n')

# --------------------------- Q 5 -----------------------------------------
print(' -----          Q5          -----')
print(" --- Principal Component Analysis (PCA) for Original Dataset ---")
print('\n')
def GetPCA(dataframe):
    scaler = StandardScaler()
    scaler.fit(dataframe)
    scaled = scaler.transform(dataframe)
    pca = PCA()
    pca.fit(scaled)
    x_pca = pca.transform(scaled)
    explained_variance = pca.explained_variance_ratio_
    return x_pca, explained_variance

def GetVar(L1,L2,col1,col2):
    newdf = pd.DataFrame(L1, columns=[col1])
    newdf[col2] = L2
    return newdf

Ls = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
pca_or, var = GetPCA(df[Ls])
print(GetVar(Ls,var,'Parameter','Variance Fraction'))
GenerateScatter(pca_or[0],pca_or[1],"Top 2 Prominent Features for Original Dataset")
GenerateScatter(pca_or[2],pca_or[3],"Least 2 Prominent Features for Original Dataset")
print(" ----------------------------------------------- ")
print('\n')
print(" --- Principal Component Analysis (PCA) for Noisy Dataset ---")
print('\n')

Ls2 = ['Sepal Length', 'Sepal width', 'Petal Length', 'Petal width']
pca_no, var2 = GetPCA(df2[Ls2])
print(GetVar(Ls2,var2,'Parameter','Variance Fraction'))
GenerateScatter(pca_no[0],pca_no[1],"Top 2 Prominent Features for Noisy Dataset")
GenerateScatter(pca_no[2],pca_no[3],"Least 2 Prominent Features for Noisy Dataset")
print(" ----------------------------------------------- ")
print('\n')
print('''Explanation:- Higher the % of Variance, Higher the % of information and less is the 
Information loss, Hence Since the Sepal Length and Sepal Width have Highervariance, it Means
that Sepal Width and Sepal Length is Better than Petal Width and Petal Length ''')
print('\n')
print(" ----------------------------------------------- ")




# ------------------------------ Q 6 ---------------------------------
print(' -----          Q6          -----')
def GetGuassianMixture(data):
    EM = GaussianMixture(n_components=2)
    scaler = StandardScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)
    
    EM.fit(scaled)
    cluster = EM.predict(scaled)
    cluster_p = EM.predict_proba(scaled)
    return cluster

def GetProbabilityDistribution(data,title):
    plt.figure(figsize=(13,6))
    sns.distplot(data,kde=True,color='skyblue',hist_kws={"linewidth": 15,'alpha':0.1})
    plt.title(title)
    return plt.show()
def GetResults(dataframe,ore):
    for sp in ['Setosa', 'Versicolor', 'Virginica']:
        val = dataframe[dataframe.Species == sp][['Petal Length', 'Petal width']]
        val2 = GetGuassianMixture(val)
        GetProbabilityDistribution(val2,f"Probability Distribution of  Gaussian Mixture Model - {sp} - {ore}")

    val = dataframe[['Petal Length', 'Petal width']]
    val2 = GetGuassianMixture(val)
    GetProbabilityDistribution(val2,f"Probability Distribution of Gaussian Mixture Model - Whole Population - {ore}")
def GetMahalanobisDistance(dataframe,Sp_name,title):
    val1 = GetGuassianMixture(dataframe[dataframe.Species == Sp_name][['Petal width']])
    val2 = GetGuassianMixture(dataframe[dataframe.Species == Sp_name][['Petal Length']])
    val11 = val1.reshape(25,2)
    val22 = val1.reshape(25,2)
    results =  cdist(val11,val22,'mahalanobis')
    GetProbabilityDistribution(results,title)

# ----------- For Original Dataset -------------------

GetResults(df3,'Original Data')
for i in ['Setosa', 'Versicolor', 'Virginica']:
    GetMahalanobisDistance(df3,i,f"Mahalanobis Distance - Original Data - {i}")

# ----------- For Noisy Dataset ---------------------

GetResults(df2,'Noisy Data')
for i in ['Setosa', 'Versicolor', 'Virginica']:
    GetMahalanobisDistance(df2,i,f"Mahalanobis Distance - Noisy Data - {i}")

# ----------------------------------------------------
print(" ----------------------------------------------- ")
print('\n')
print('Explanation:- We Have seen throught Various Matrics on the Data and we can see clearly, those two dataset are very identical in nature.')
print('\n')
print(" ----------------------------------------------- ")


# --------------------- Q 7 -------------------------------
print(' -----          Q7          -----')
def GetResults2(dataframe,ore):
    val = dataframe[['Petal Length', 'Petal width']]
    val2 = GetGuassianMixture(val)
    GetProbabilityDistribution(val2,f"Probability Distribution of Gaussian Mixture Model - Principal Componenet - {ore}")

p_comps1 = pd.DataFrame(pca_or,columns=['Sepal Length','Sepal width','Petal Length','Petal width'])
GetResults2(p_comps1,'Original Dataset')

p_comps2 = pd.DataFrame(pca_no,columns=['Sepal Length','Sepal width','Petal Length','Petal width'])
GetResults2(p_comps2,'Noisy Dataset')






