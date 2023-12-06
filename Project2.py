# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:35:24 2023
@author: Steve
"""

import prince
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
from NN import draw_network, create_network
from sklearn.metrics import confusion_matrix
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

df=pd.read_csv('heart.csv')
df_en=df.copy()
########################################################################### Preprocessing
df_en.Sex.replace({'M':1,'F':0},inplace=True)
df_en.ChestPainType.replace({'ATA':0,'NAP':1,'ASY':2,'TA':3},inplace=True)
df_en.RestingECG.replace({'Normal':0,'ST':1,'LVH':2},inplace=True)
df_en.ExerciseAngina.replace({'Y':1,'N':0},inplace=True)
df_en.ST_Slope.replace({'Up':0,'Flat':1,'Down':2},inplace=True)
df_en.Cholesterol.replace({0:np.NaN},inplace=True)
df_en.RestingBP.replace({0:np.NaN},inplace=True)

scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df_en), columns = df_en.columns)

imputer = KNNImputer(n_neighbors=5)
df3 = pd.DataFrame(imputer.fit_transform(df2),columns = df2.columns)

df4=pd.DataFrame(scaler.inverse_transform(df3),columns=df2.columns)

df5=df4.copy()

df5.Sex.replace({1:'M',0:'F'},inplace=True)
df5.ChestPainType.replace({0:'ATA',1:'NAP',2:'ASY',3:'TA'},inplace=True)
df5.RestingECG.replace({0:'Normal',1:'ST',2:'LVH'},inplace=True)
df5.ExerciseAngina.replace({1:'Y',0:'N'},inplace=True)
df5.ST_Slope.replace({0:'Up',1:'Flat',2:'Down'},inplace=True)
orig=df.copy()
df=df4.copy()
imp_vars=['Age','MaxHR','Oldpeak','Sex','ChestPainType','FastingBS','ExerciseAngina','ST_Slope']

############################################################################  Dimension reduction
famd = prince.FAMD(n_components=2, n_iter=3, copy=True, check_input=True,random_state=42)
res=famd.fit_transform(df5[imp_vars])
res.columns=['x','y']

# dist_matrix=gower.gower_matrix(df5[imp_vars])
# umap_embeddings=umap.UMAP(random_state=0, n_components=2).fit_transform(dist_matrix)
# res=pd.DataFrame(umap_embeddings,columns=['x','y'])

X= np.matrix(res)
x_min, x_max = X[:, 0].min() - 1.5, X[:,0].max() + 1.5
y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3
xrange=np.linspace(x_min,x_max, 100)
yrange=np.linspace(y_min, y_max, 100)
xx, yy = np.meshgrid(np.linspace(x_min,x_max, 100),
np.linspace(y_min, y_max, 100))
x_in = np.c_[xx.ravel(), yy.ravel()]

############################################################################  Train test splits

xtrain,xtest,ytrain,ytest= train_test_split(df5[imp_vars],df.HeartDisease,test_size=0.2,random_state=7)

famd2=prince.FAMD(n_components=2, n_iter=3, copy=True, check_input=True,random_state=42)

xtrain=famd2.fit_transform(xtrain)
xtest=famd2.transform(xtest)

xtrain.columns=xtest.columns=['x','y']

def reduce(training_set, testing_set):
    famd_gen = prince.FAMD(n_components=2, n_iter=3, copy=True, check_input=True,random_state=42)
    train = famd_gen.fit_transform(training_set)
    test = famd_gen.transform(testing_set)
    train.columns=test.columns=['x','y']
    return train,test


#############################################################################  Plotting function

def plot(clf, x, y, prob=False, threshold=None, train=False, Final=False):
    
    if prob and not threshold:
        Z = clf.predict_proba(x_in)[:, 1]
    
    elif prob and threshold:
        Z = clf.predict_proba(x_in)[:, 1]
        Z=(Z>=threshold)*1
        prob=False
    
    else:
        Z = clf.predict(x_in)
        
    Z = Z.reshape(xx.shape)
    
    if prob:
        c=px.colors.sequential.Jet
    else:
        if Z.min() == 0 and Z.max()==1:
            c=[[0,'#85cc18'],[1,'#dc143c']]
        elif Z.min()==1 and Z.max()==1:
            c=[[0,'#dc143c'],[1,'#dc143c']]
        elif Z.min()==0 and Z.max()==0:
            c=[[0,'#85cc18'],[1,'#85cc18']]
        
        legend_markers = [
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color='#85cc18', symbol='square', size=10, opacity=0.5),
                name='0',
                legend='legend2'
            ),
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color='#dc143c', symbol='square', size=10, opacity=0.5),
                name='1',
                legend='legend2'
            )
        ]
        
    if train:
        title='Train set'
    else:
        title='Test set'
        
    fig1 = go.Figure(data=[
        go.Contour(
            x=xrange,
            y=yrange,
            z=Z,
            colorscale=c,
            opacity=0.5,
            showscale=prob,
            colorbar={'title':'Probability of Heart Disease', 'title_font_size':12}
        )
    ])

    fig2= px.scatter(x=x.x,y=x.y,color=y.astype(int).astype(str), 
                     symbol_sequence=['x'],
                     color_discrete_map= {'1': '#dc143c',
                                                       '0': '#85cc18'})
    fig2.update_traces(marker_size=12,marker_opacity=0.7, showlegend=True, marker_line=dict(width=1,color='DarkSlateGrey')) 
    
    if Final:
        fig2.update_traces(marker_size=15,marker_opacity=1, showlegend=True, marker_line=dict(width=3,color='Black'))
        fig1.update_traces(opacity=0.4)
    
    
    fig=go.Figure(data=fig1.data+fig2.data)
    
    if not prob:
        for marker_trace in legend_markers:
            fig.add_trace(marker_trace)
    
    ## Updating overall layout
    ## plot_bgcolor="white",
    fig.update_layout(title_text=title, title_x=0.5, title_font_size= 20, title_y=0.85, height=500, width=900,
                      xaxis_title='Dimension 1', yaxis_title='Dimension 2')

    ## Updating legend layout
    fig.update_layout(legend=dict(yanchor="top", xanchor="left", x=-0.3, y=0.975, title='Truth Labels'),
                      legend2=dict(title='Prediction Area', y=0.975, title_font_size=12.75, x=1.05))
    
        
    return fig

###########################################################################  Writing metrics function

def strc(acc,prec,recall,f1,data, hor=False):
    if acc>=0.8:
        cacc='green'
    else:
        cacc='red'
    if prec>=0.8:
        cprec='green'
    else:
        cprec='red'
    if recall>=0.8:
        crecall='green'
    else:
        crecall='red'
    if f1>=0.8:
        cf1='green'
    else:
        cf1='red'
    
    if hor:
        
        return f"""#### {data.capitalize()} Set Metrics:
##### :{cacc}[Accuracy : {round(acc*100,2)}%] , :{cprec}[Precision : {round(prec*100,2)}%] , :{crecall}[Recall : {round(recall*100,2)}%] , :{cf1}[F1 score : {round(f1*100,2)}%]
    """
    else:
    
        return f"""#### {data.capitalize()} Set Metrics:
#### :{cacc}[Accuracy : {round(acc*100,2)}%]\n
#### :{cprec}[Precision : {round(prec*100,2)}%]\n
#### :{crecall}[Recall : {round(recall*100,2)}%]\n
#### :{cf1}[F1 score : {round(f1*100,2)}%]
    """
    
with st.sidebar:
    selected = option_menu("Content", ["Overview", "Dimensionality reduction","Logistic Regression","KNN Classifier",'Decision Tree',
                                       'Random Forest', "SVM","Neural Networks","Summary"],
                          default_index=0,
                          orientation="vertical",
                          styles={"nav-link": {"font-size": "15px", "text-align": "centre", "margin": "0px", 
                                                "--hover-color": "#85cc18"},
                                   "icon": {"font-size": "15px"},
                                   "container" : {"max-width": "3500px"},
                                   "nav-link-selected": {"background-color": "#e01631"}})


if selected=='Overview':
    
    sec1, sec2, sec3, data_info , sec4, sec5 = st.tabs(['Goal', 'App features' , 'Metrics', 'Data Info' ,'About Me' ,'References'])
    
    img=Image.open('title.jpg')
    sec1.image(img)
    
    sec1.markdown("""
                  ## What is the goal?
                  
                  - The goal is to model the relationships between different variables and heart disease and create a customized
                  model to predict heart disease
                  - In the **Summary** section, we'll be predicting for future datasets. We'll go through some models to get the best 
                  possible model for our usecase

                  ## Why Machine Learning?
                  
                  - It is tedious to learn the relationship between all the 8 variables and heart disease manually. Even if we do, it is 
                  difficult to predict heart disease (Tons of manual calculations will be involved!). So we use Machine learning! :computer:
                      
                  ## Can we use any random model?
                  
                  - **Since heart disease is a sensitive issue, it is not advisable to do that.**
                  - In this app, we will start out from simple models like logistic regression to complicated models like neural networks
                  - We'll try out different models and hyperparameters. Finally, we will select the model we like and predict for 
                  future data in the 'Summary' section
                  
                  
                  """)
    
    sec1.markdown("""
                  ## Shouldn't we start with EDA?
                  
                  - **YES!** Please go to this [link to see the EDA](https://heart-disease-project-steve.streamlit.app/)
                  
                  We have identified 8 different variables as important factors for predicting heart disease. They are:
                      
                  **Age, MaxHR, Oldpeak, Sex, ChestPainType, FastingBS, ExerciseAngina, and ST_Slope**
                  
                  See the image below for a sample of the EDA app!
                  
                  """)
    
    img=Image.open('EDA.jpg')
    sec1.image(img,caption='EDA App Sample')
                  
    sec1.markdown("""
                  ## Classification or Regression?
    
                  """)
    img=Image.open('regvsclass.jpg')
    sec1.image(img)
    
    sec1.markdown("""
                  - Since we are predicting heart disease based on previous labels, it is supervised learning. Specifically, since
                  we are predicting discrete classes (Whether a person has heart disease or not), we are using classification models.
                  """)
    
    sec1.caption("""
                  Image credits:
                - [Machine Mantra](https://machinemantra.in/heart-disease-prediction-in-python/)
                - [Simplilearn](https://www.simplilearn.com/regression-vs-classification-in-machine-learning-article)
                  """)
    
    sec2.write('## Let us look at the different features:')
    img=Image.open('features.png')
    sec2.image(img)
    
    sec3.markdown("""
                  ## Let us review some metrics we use throughout:
                  #### The confusion matrix: """)
    img=Image.open('conf.jpg')
    sec3.image(img)
    
    sec3.markdown("""
                  - **Accuracy** is an important metric for classification. It is the ratio of correct predictions to the total predictions
                  - You can see the number of right and wrong predictions in the confusion matrix
                  - Ideally we want our model to predict with 100% accuracy. But, in the real world that is not possible. So, we must choose 
                  the type of errors we want to avoid. 
                  
                  See below for how to prevent those errors
                  """)
    
    sec3.write('## Precision vs Recall')
    
    img = Image.open('prerec.png')
    sec3.image(img)
    
    sec3.markdown("""
                  - High **precision** ensures we prevent False positives
                  - High **recall** ensures we prevent False negatives
                  - You can modify the model's precision and recall in the **'Precision-Recall Tradeoff'** section!
                  
                  **Note:** F1 score is nothing but the harmonic mean of precision and recall
                  """)
    sec3.caption("""
                  Image credits:
                - [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/)
                - [towardsdatascience](https://towardsdatascience.com/precision-and-recall-made-simple-afb5e098970f)
                  """)
                  
    data_info.markdown("""
    The UCI Heart Disease dataset contains various clinical and demographic variables associated with heart disease. Below is a detailed overview of the dataset variables:
    
    ### 1. **Age:**
    - **Description:** Age of the patient.
    - **Type:** Continuous.
    
    ### 2. **Sex:**
    - **Description:** Gender of the patient.
    - **Type:** Categorical (F = female, M = male).
    
    ### 3. **ChestPainType :**
    - **Description:** Type of chest pain experienced by the patient.
    - **Type:** Categorical (TA, ATA, NAP, ASY).
    - **Categories:**
      - TA: Typical angina.
      - ATA: Atypical angina.
      - NAP: Non-anginal pain.
      - ASY: Asymptomatic.
    
    ### 4. **RestingBP:**
    - **Description:** Resting blood pressure of the patient (in mm Hg).
    - **Type:** Continuous.
    
    ### 5. **Cholesterol:**
    - **Description:** Serum cholesterol level of the patient (in mg/dl).
    - **Type:** Continuous.
    
    ### 6. **FastingBS:**
    - **Description:** Fasting blood sugar level of the patient.
    - **Type:** Categorical (0 = blood sugar < 120 mg/dl, 1 = blood sugar > 120 mg/dl).
    
    ### 7. **RestingECG:**
    - **Description:** Resting electrocardiographic measurement results.
    - **Type:** Categorical (Normal, ST, LVH).
    - **Categories:**
      - Normal: Normal.
      - ST: Abnormality related to ST-T wave.
      - LVH: Hypertrophy of the left ventricle.
    
    ### 8. **MaximumHR:**
    - **Description:** Maximum heart rate achieved during exercise.
    - **Type:** Continuous.
    
    ### 9. **ExerciseAngina:**
    - **Description:** Exercise-induced angina (chest pain) observed.
    - **Type:** Categorical (N = no, Y = yes).
    
    ### 10. **Oldpeak:**
    - **Description:** ST depression induced by exercise relative to rest.
    - **Type:** Continuous.
    
    ### 11. **ST_Slope:**
    - **Description:** Slope of the peak exercise ST segment.
    - **Type:** Categorical (Up, Flat, Down).
    - **Categories:**
      - Up: Upsloping.
      - Flat: Flat.
      - Down: Downsloping.
    
    ### 14. **HeartDisease:**
    - **Description:** Presence or absence of heart disease.
    - **Type:** Categorical (0 = no heart disease, 1 = heart disease).
    
    Researchers and data scientists use this dataset to analyze factors contributing to heart disease and build predictive models for early detection and prevention.
    """)
                  
    sec4.markdown("""
                  ## Hello!
                  """)
    _,middle,_=sec4.columns(3)
    dp=Image.open('DP.jpg')
    middle.image(dp,width=400)
    sec4.markdown("""
                  I am Steve Mitchell, a driven Data Science enthusiast currently pursuing a Master's in Data Science from Michigan 
                  State University. With a background in shaping data strategies for business growth, I have a knack for translating 
                  complex data into actionable insights. My expertise lies in **Python, SQL, and machine learning**. 
                  I am passionate about leveraging data to **predict customer behavior, churn, fraud, and building data-driven dashboards**
                  to improve the business. I am eager to contribute to innovative teams, utilizing proactive stakeholder engagement 
                  and data-driven decision-making skills to spearhead projects.
                  I am excited about driving positive change through technology and am ready to bring valuable insights to your team!
                  
                  Visit my linkedin page for more details!
                  
                  [Linkedin](https://www.linkedin.com/in/stevemitchellr/)
                  """)
                  
    sec5.markdown("""
                  ## References
                  - [Medium](https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b)
                  - [Plotly](https://plotly.com/python/line-and-scatter/#scatter-and-line-plots-with-goscatter)
                  - [monkeylearn](https://monkeylearn.com/blog/classification-algorithms/)
                  - [machinelearningmastery](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)
                  """)

if selected=='Dimensionality reduction':
    
    sec1, sec2 = st.tabs(['What is FAMD?','FAMD visualization'])
    
    sec1.markdown("""
                  ## FAMD
                  FAMD (Factor Analysis of Mixed Data) is a dimensionality reduction algorithm which reduces a multi-dimensional 
                  dataset into lower dimensions. It is similar to PCA.
                  
                  ## Why do we want to reduce dimensions?
                  - :brain: **Less memory consumption**
                  - :chart_with_upwards_trend: **Faster results** (Especially when we want to experiment with a lot of options like in this app!)
                  - :bar_chart: **Powerful visualizations** (Can help us understand and fine tune our model by visualizing how it is classifying the points
                                             and understanding the decision boundaries better! You can see the power of visualization
                                             in the next sections)
                  
                  ## Why not PCA?
                  
                  - PCA tries to capture the direction of maximum variance in the dataset. But variance is not a property of categorical
                  variables. So, we switch to FAMD which can reduce dimensions for both quantitative and categorical variables.
                  - Basically FAMD is an upgraded version of PCA
                  
                  Take a look at the picture below for an example of dimension reduction!
                  """)
    image= Image.open('FAMD.png')
    sec1.image(image, width=900)
    
    sec1.markdown("""
                  - FAMD can reduce the dataset from **n-dimensions to 2 dimensions!** :small_red_triangle_down:
                  
                  - FAMD finds the distances between points in n-dimensions and retains these relationships when mapped onto a lower dimension 
                  (2 dimensions in this case)
                  
                  - Switch to the next tab to see the application of FAMD to our dataset!
                  """)
    sec1.caption("""
                 Image credits:
                - [researchgate](https://www.researchgate.net/figure/Dimensionality-reduction-effect-over-an-3D-artificial-Swiss-roll-manifold-the-2D_fig1_325363944)
                 """)
    
    sec2.markdown("""
                  ## How many features do we have?
                  
                  We have identified 8 features as important factors of heart disease. 
                  
                  **HOW?** -  refer to this link where we have 
                  explored the dataset and identified the important features: [EDA](https://heart-disease-project-steve.streamlit.app/)
                  
                  Unfortunately, we cannot plot all the features in 8 dimensions. But we can see the results after applying FAMD below:
                  """)
    fig= px.scatter(x=res.x,y=res.y,color=df.HeartDisease.astype(int).astype(str), 
                      symbol_sequence=['x'],
                      color_discrete_map= {'1': '#dc143c',
                                                        '0': '#85cc18'})
    fig.update_traces(marker_size=12,marker_opacity=0.7, showlegend=True, marker_line=dict(width=1,color='DarkSlateGrey')) 
    fig.update_layout(title_text='UMAP dataset', title_x=0.4, title_font_size= 20, title_y=0.95, height=500, width=800,
                      xaxis_title='Dimension 1', yaxis_title='Dimension 2')
    fig.update_layout(legend=dict(title='Truth Labels'))
    
    sec2.plotly_chart(fig)
    
    sec2.markdown("""
                  FAMD has identified the relationship between points that are in 8 dimensions! These relationships that are formed across
                  8 dimensions are preserved in 2 dimensions using FAMD.
                  """)
    

###########################################################    Models    ##########################################################

if selected=='Logistic Regression':
    
    sec0,sec1,sec2,sec3 = st.tabs(['Overview','Predictions','Probabilities','Precision-Recall tradeoff'])
    
    sec0.markdown("""
                  ## Motivation
                  
                  Let's start with a very basic model. Logistic regression is the most basic model and is easy to interpret and implement.
                  It tries to classify the points using a straight line. 
                  
                  Let's see more details about it!
                  
                  ## What is Logistic Regression?
                  """)
    img=Image.open('log.png')
    sec0.image(img)
    
    sec0.markdown("""
                  - Logistic regression is a **linear** classifier. The reason for this is it has linear regression in the background and it
                  creates a linear decision boundary to separate the classes.
                  - The S-shaped curve is observed when we have the target in the Y-axis and the feature in the X-axis. But, when we plot multiple
                  features and apply logistic regression, a linear decision boundary is observed.
                  """)
                  
    sec0.caption("""
                 Image credits:
                - [spiceworks](https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/)
                 """)
    
    sec1.markdown("""
    #### Overview: 
    - The **solvers** determine how we arrive at the solution. Try out different solvers and see how it affects the decision boundaries!
    - **Penalties** prevent overfitting (Fitting to noise). Experiment with the penalty strength and type!
    
    **Try out the CUSTOM DATA SET feature to define your train and test sets!**
                """)
    
    
    val=sec1.checkbox('USE CUSTOM DATA SET',value=False,key='a')
    if val:
        lefty, righty = sec1.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='1')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-1')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
    
    left,mid,right=sec1.columns(3)
    
    solver=left.selectbox('Choose solver:', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],key='11')
    
    if solver=='newton-cg' or solver=='lbfgs' or solver=='sag':
        pen=['l2','none']
    elif solver=='liblinear':
        pen=['l1','l2']
    else:
        pen=['none','l1','l2','elasticnet']
        
    
    penalty=mid.selectbox('Choose the penalty',pen,key='111')
    if penalty=='none':
        penalty=None
    
    C=right.selectbox('Choose penalty strength',[100,10,1,0.1,0.01],key='1111')
    
    model=LogisticRegression(solver=solver,penalty=penalty,C=C)
    model.fit(xtrain,ytrain)
    
    fig_train=plot(model,xtrain,ytrain,train=True)
    sec1.plotly_chart(fig_train,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    

    fig_test=plot(model,xtest,ytest)
    sec1.plotly_chart(fig_test,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    ##################################   Probabilities                 ###########################################################
    
    sec2.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab
                - Adjust different settings to change how likely the model predicts someone might have heart disease
                  """)
    
    val=sec2.checkbox('USE CUSTOM DATA SET',value=False,key='b')

    if val:
        lefty, righty = sec2.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='2')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-2')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
    
    left,mid,right=sec2.columns(3)
    
    solver=left.selectbox('Choose solver:', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],key='22')
    
    if solver=='newton-cg' or solver=='lbfgs' or solver=='sag':
        pen=['l2','none']
    elif solver=='liblinear':
        pen=['l1','l2']
    else:
        pen=['none','l1','l2','elasticnet']
        
    
    penalty=mid.selectbox('Choose the penalty',pen,key='222')
    if penalty=='none':
        penalty=None
    
    C=right.selectbox('Choose penalty strength',[1,100,10,0.1,0.01],key='2222')
    
    model=LogisticRegression(solver=solver,penalty=penalty,C=C)
    model.fit(xtrain,ytrain)
    
    fig_train2=plot(model,xtrain,ytrain,prob=True,train=True)
    sec2.plotly_chart(fig_train2,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    
    
    fig_test2=plot(model,xtest,ytest,prob=True)
    sec2.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    #############################################    Precision - Recall ##########################################
    
    sec3.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab. 
                - **Normally, models predict 1 if the probability is over 0.5 (default probability threshold). Customize your model by picking 
                the probability threshold!**
                - A high threshold means the model says someone has heart disease **ONLY** if it's 
                absolutely sure!
                - A small threshold makes the model **MORE CAUTIOUS**—it says someone has 
                heart disease even if it's just a little suspicious! **Choose what you want for your problem**
                  """)
    
    val=sec3.checkbox('USE CUSTOM DATA SET',value=False,key='c')
    if val:
        lefty, righty = sec3.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='3')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-3')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
    
    left,mid,right=sec3.columns(3)
    
    solver=left.selectbox('Choose solver:', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],key='33')
    
    if solver=='newton-cg' or solver=='lbfgs' or solver=='sag':
        pen=['l2','none']
    elif solver=='liblinear':
        pen=['l1','l2']
    else:
        pen=['none','l1','l2','elasticnet']
        
    
    penalty=mid.selectbox('Choose the penalty',pen,key='333')
    if penalty=='none':
        penalty=None
    
    C=right.selectbox('Choose penalty strength',[100,10,1,0.1,0.01],key='3333')
    threshold=sec3.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='33333')
    
    model=LogisticRegression(solver=solver,penalty=penalty,C=C)
    model.fit(xtrain,ytrain)

    fig_test2=plot(model,xtest,ytest,prob=True, threshold=threshold)
    sec3.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=(model.predict_proba(xtest)[:,1]>=threshold)*1.0
    
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    
    left2,right2=sec3.columns(2)
    
    left2.markdown(strc(acc,prec,recall,f1,'test'))
    sns.set(font_scale = 1.5)
    fig=sns.heatmap(confusion_matrix(ytest,ypreds),annot=True,cmap='viridis',fmt='.0f', annot_kws={'size': 20})
    fig.set(ylabel='True labels', xlabel='Predictions', title='Confusion matrix')
    right2.pyplot()
    
if selected=='Decision Tree':
    
    sec0,sec1,sec2,sec3 = st.tabs(['Overview','Predictions','Probabilities','Precision-Recall tradeoff'])
    
    sec0.markdown("""
                  
                  ## Motivation
                  
                  Now, on to some more powerful models. A decision tree splits the entire dataset into different sections and classifies the
                  data. So, it is not limited by the assumption of KNN. It can also classifies similar points that are farther apart.
                  Let's see more details about it!
                  
                  ## What is a Decision Tree?
                  Let's take a look at an example! Observe that the tree is classifying whether it's a dog or an elephant based on
                  certain conditions
                  """)
    img=Image.open('Decision tree.jpg')
    sec0.image(img)
    sec0.markdown("""
                  - As the name suggests, decision trees take decisions to classify data based on conditions (For example, is age>40 or not?)
                  - Since the decisions are concrete, they do not have a probability associated with them. (Test it out in 
                  the 'Probabilities' section)
                  - Decision trees are non-linear classifiers and can work with non-linear data pretty well
                  """)
    sec0.caption("""
                 Image credits:
                - [ML/DS course](https://machine-learning-and-data-science-with-python.readthedocs.io/en/latest/assignment5_sup_ml.html)
                 """)
    
    
    sec1.markdown("""
    #### Overview: 
    - Modify the height, min samples split, and the min samples leaf to prevent overfitting (Fitting to noise)
    - The **height** determines the height of the tree.
    - **Min Samples Split** decides how many samples a group needs to split more, while **Min Samples Leaf** decides how many samples 
    can be in the final groups.
    
    **Try out the CUSTOM DATA SET feature to define your train and test sets!**
                """)
    
    
    val=sec1.checkbox('USE CUSTOM DATA SET',value=False,key='a')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec1.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='1')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-1')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
        
    max_depth=sec1.slider('Choose the height of the tree:', min_value=5, max_value=30, value=30 ,key='11')
    
    left,right=sec1.columns(2)
    min_samples_split=left.selectbox('Choose the minimum samples to split a node',[2,5,10,15,20,30,50],key='111')
    min_samples_leaves=right.selectbox('Choose the minimum samples in a leaf node',[1,5,10,15,20,30,50],key='1111')
    
    model=tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves)
    model.fit(xtrain,ytrain)
    
    fig_train=plot(model,xtrain,ytrain,train=True)
    sec1.plotly_chart(fig_train,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    

    fig_test=plot(model,xtest,ytest)
    sec1.plotly_chart(fig_test,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    
    ##################################   Probabilities                 ###########################################################
    
    sec2.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab
                - Adjust different settings to change how likely the model predicts someone might have heart disease
                  """)
    
    val=sec2.checkbox('USE CUSTOM DATA SET',value=False,key='b')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec2.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='2')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-2')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    max_depth=sec2.slider('Choose the height of the tree:', min_value=5, max_value=30, value=30 ,key='22')
    
    left,right=sec2.columns(2)
    
    min_samples_split=left.selectbox('Choose the minimum samples to split a node',[2,5,10,15,20,30,50],key='222')
    min_samples_leaves=right.selectbox('Choose the minimum samples in a leaf node',[1,5,10,15,20,30,50],key='2222')
    
    model=tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves)
    model.fit(xtrain,ytrain)
    
    fig_train2=plot(model,xtrain,ytrain,prob=True,train=True)
    sec2.plotly_chart(fig_train2,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    
    fig_test2=plot(model,xtest,ytest,prob=True)
    sec2.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    #############################################    Precision - Recall ##########################################
    
    sec3.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab. 
                - **Normally, models predict 1 if the probability is over 0.5 (default probability threshold). Customize your model by picking 
                the probability threshold!**
                - A high threshold means the model says someone has heart disease **ONLY** if it's 
                absolutely sure!
                - A small threshold makes the model **MORE CAUTIOUS**—it says someone has 
                heart disease even if it's just a little suspicious! **Choose what you want for your problem**
                  """)
    
    val=sec3.checkbox('USE CUSTOM DATA SET',value=False,key='c')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec3.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='3')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-3')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left,right=sec3.columns(2)
   
    min_samples_split=left.selectbox('Choose the minimum samples to split a node',[2,5,10,15,20,30,50],key='333')
    min_samples_leaves=right.selectbox('Choose the minimum samples in a leaf node',[1,5,10,15,20,30,50],key='3333')
   
    threshold=sec3.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='33333')
    
    model=tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves)
    model.fit(xtrain,ytrain)

    fig_test2=plot(model,xtest,ytest,prob=True, threshold=threshold)
    sec3.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=(model.predict_proba(xtest)[:,1]>=threshold)*1.0
    
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    
    left2,right2=sec3.columns(2)
    
    left2.markdown(strc(acc,prec,recall,f1,'test'))
    sns.set(font_scale = 1.5)
    fig=sns.heatmap(confusion_matrix(ytest,ypreds),annot=True,cmap='viridis',fmt='.0f', annot_kws={'size': 20})
    fig.set(ylabel='True labels', xlabel='Predictions', title='Confusion Matrix')
    right2.pyplot()
    
if selected=='Random Forest':
    sec0,sec1,sec2,sec3 = st.tabs(['Overview','Predictions','Probabilities','Precision-Recall tradeoff'])
    
    sec0.markdown("""
                  
                  ## Motivation
                  
                  If one decision tree is good, more is better! A random forest utilizes the principle of using multiple models and so is
                  much more powerful than a single decision tree. It improves upon decision trees.
                  
                  Let's see more details about it!
                  
                  """)
    
    img=Image.open('RF.jpg')
    sec0.image(img, width=700)
    
    sec0.markdown("""
                  ## What is a Random Forest?
                  - A random forest is a collection of many decision trees, each of which uses a different subset of data.
                  - The decision trees' results are combined through voting to predict the final outcome.
                  
                  Take a look at the image below!
                  """)
    
    img2=Image.open('Random Forest.jpg')
    sec0.image(img2)
    
    sec0.write('- Here, most of the trees predict 1 and so the prediction by the Random Forest is 1')
    
    sec0.caption("""
                 Image credits:
                - [fromthegenesis](https://www.fromthegenesis.com/random-forest-classification-r/)
                - [towardsdatascience](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
                 """)
    
    sec1.markdown("""
    #### Overview: 
    - Modify the height, min samples split, and the min samples leaf to prevent overfitting (Fitting to noise)
    - The **height** determines the height of the tree. Reducing the height prevents overfitting
    - **Min Samples Split** sets the samples needed to split a group, while **Min Samples Leaf** determines the minimum samples 
    allowed in final groups. Raising these helps avoid overfitting.
    - Increasing the **number of estimators** means using more trees to make better predictions. But it will cost you time
    
    **Try out the CUSTOM DATA SET feature to define your train and test sets!**
                """)
    
    val=sec1.checkbox('USE CUSTOM DATA SET',value=False,key='a')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec1.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='1')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-1')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    max_depth=sec1.slider('Choose the height of the tree:', min_value=5, max_value=30, value=30 ,key='11')
    
    left,mid,right=sec1.columns(3)
    

    min_samples_split=left.selectbox('Min samples to split a node',[2,5,10,15,20,30,50],key='111')
    min_samples_leaves=mid.selectbox('Min samples in a leaf node',[1,5,10,15,20,30,50],key='1111')
    n_estimators=right.selectbox('Choose the number of estimators',[201,101,51,31,11,1],key='11111')
    
    model=RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves,
                                 n_estimators=n_estimators)
    model.fit(xtrain,ytrain)
    
    fig_train=plot(model,xtrain,ytrain,train=True)
    sec1.plotly_chart(fig_train,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    

    fig_test=plot(model,xtest,ytest)
    sec1.plotly_chart(fig_test,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    ##################################   Probabilities                 ###########################################################
    
    sec2.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab
                - Adjust different settings to change how likely the model predicts someone might have heart disease
                  """)
    
    val=sec2.checkbox('USE CUSTOM DATA SET',value=False,key='b')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec2.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='2')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-2')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    max_depth=sec2.slider('Choose the height of the tree:', min_value=5, max_value=30, value=30 ,key='22')
    
    left,mid,right=sec2.columns(3)
    
    min_samples_split=left.selectbox('Min samples to split a node',[2,5,10,15,20,30,50],key='222')
    min_samples_leaves=mid.selectbox('Min samples in a leaf node',[1,5,10,15,20,30,50],key='2222')
    n_estimators=right.selectbox('Choose the number of estimators',[201,101,51,31,11,1],key='22222')
    
    model=RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves,
                                 n_estimators=n_estimators)
    model.fit(xtrain,ytrain)
    
    fig_train2=plot(model,xtrain,ytrain,prob=True,train=True)
    sec2.plotly_chart(fig_train2,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    
    fig_test2=plot(model,xtest,ytest,prob=True)
    sec2.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    #############################################    Precision - Recall ##########################################
    
    sec3.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab. 
                - **Normally, models predict 1 if the probability is over 0.5 (default probability threshold). Customize your model by picking 
                the probability threshold!**
                - A high threshold means the model says someone has heart disease **ONLY** if it's 
                absolutely sure!
                - A small threshold makes the model **MORE CAUTIOUS**—it says someone has 
                heart disease even if it's just a little suspicious! **Choose what you want for your problem**
                  """)
    
    val=sec3.checkbox('USE CUSTOM DATA SET',value=False,key='c')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec3.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='3')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-3')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    max_depth=sec3.slider('Choose the height of the tree:', min_value=5, max_value=30, value=30 ,key='33')
    
    left,mid,right=sec3.columns(3)
    
    min_samples_split=left.selectbox('Min samples to split a node',[2,5,10,15,20,30,50],key='333')
    min_samples_leaves=mid.selectbox('Min samples in a leaf node',[1,5,10,15,20,30,50],key='3333')
    n_estimators=right.selectbox('Choose the number of estimators',[201,101,51,31,11,1],key='33333')
    
   
    threshold=sec3.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
    
    model=RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves,
                                 n_estimators=n_estimators)
    model.fit(xtrain,ytrain)

    fig_test2=plot(model,xtest,ytest,prob=True, threshold=threshold)
    sec3.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=(model.predict_proba(xtest)[:,1]>=threshold)*1.0
    
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    
    left2,right2=sec3.columns(2)
    
    left2.markdown(strc(acc,prec,recall,f1,'test'))
    sns.set(font_scale = 1.5)
    fig=sns.heatmap(confusion_matrix(ytest,ypreds),annot=True,cmap='viridis',fmt='.0f', annot_kws={'size': 20})
    fig.set(ylabel='True labels', xlabel='Predictions', title='Confusion Matrix')
    right2.pyplot()
    
if selected=='KNN Classifier':
    sec0,sec1,sec2,sec3 = st.tabs(['Overview','Predictions','Probabilities','Precision-Recall tradeoff'])
    
    sec0.markdown("""
                  
                  ## Motivation
                  
                  Let's move on to a slightly advanced algorithm. KNN classifier assumes that points of the same class are closer together.
                  This may or may not be true for different datasets. 
                  
                  Let's see more details about it!
                  
                  ## What is KNN?
                  
                  - K Nearest Neighbors algorithm assumes that points that are similar are closer together
                  - It sees K points that are closest to the data point and assigns a class based on the class of the closest points
                  
                  See the image below!
                  """)
    img = Image.open('KNN.png')
    sec0.image(img)
    
    sec0.write("""- **Out of 3 closest points (k=3), 2 points belong to class B and one point belongs to class A. So, the new data 
               point is assigned to class B based on majority**""")
    sec0.caption("""
                 Image credits:
                - [kdnuggets](https://www.kdnuggets.com/2022/04/nearest-neighbors-classification.html)
                 """)
    
    sec1.markdown("""
    #### Overview: 
    - The **neighbors count** decides how many close points to look at when labeling a new point. Too few could overfit (cling to noise), 
    while too many might underfit (miss patterns in data).
    - When using **uniform weights**, all nearby neighbors carry equal importance. But with **distance-based weights**, the closer points 
    hold greater significance in classifying a new point.
    - **Algorithm** determines how we calculate the nearest neighbors. You can also choose the **type of distance** you want to calculate between two points
    
    **Try out the CUSTOM DATA SET feature to define your train and test sets!**
                """)
    
    val=sec1.checkbox('USE CUSTOM DATA SET',value=False,key='a')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec1.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='1')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-1')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    n_neighbors=sec1.slider('Choose the number of neighbors:', min_value=5, max_value=30, value=5, step=5 ,key='11')
    
    left,mid,right=sec1.columns(3)
    
    weights=left.selectbox('Select the weights function',['uniform','distance'],key='111')
    algorithm=mid.selectbox('Select the algorithm',['auto','ball_tree','kd_tree','brute'],key='1111')
    metric=right.selectbox('Choose the type of distance',['minkowski','euclidean','manhattan'],key='11111')
    
    model=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)
    model.fit(xtrain,ytrain)
    
    fig_train=plot(model,xtrain,ytrain,train=True)
    sec1.plotly_chart(fig_train,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    

    fig_test=plot(model,xtest,ytest)
    sec1.plotly_chart(fig_test,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    ##################################   Probabilities                 ###########################################################
    
    sec2.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab
                - Adjust different settings to change how likely the model predicts someone might have heart disease
                  """)
    
    val=sec2.checkbox('USE CUSTOM DATA SET',value=False,key='b')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec2.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='2')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-2')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    n_neighbors=sec2.slider('Choose the number of neighbors:', min_value=5, max_value=30, value=5, step=5 ,key='22')
    
    left,mid,right=sec2.columns(3)
    
    weights=left.selectbox('Select the weights function',['uniform','distance'],key='222')
    algorithm=mid.selectbox('Select the algorithm',['auto','ball_tree','kd_tree','brute'],key='2222')
    metric=right.selectbox('Choose the type of distance',['minkowski','euclidean','manhattan'],key='22222')
    
    model=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)

    model.fit(xtrain,ytrain)
    
    fig_train2=plot(model,xtrain,ytrain,prob=True,train=True)
    sec2.plotly_chart(fig_train2,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    
    fig_test2=plot(model,xtest,ytest,prob=True)
    sec2.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    #############################################    Precision - Recall ##########################################
    
    sec3.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab. 
                - **Normally, models predict 1 if the probability is over 0.5 (default probability threshold). Customize your model by picking 
                the probability threshold!**
                - A high threshold means the model says someone has heart disease **ONLY** if it's 
                absolutely sure!
                - A small threshold makes the model **MORE CAUTIOUS**—it says someone has 
                heart disease even if it's just a little suspicious! **Choose what you want for your problem**
                  """)
    
    val=sec3.checkbox('USE CUSTOM DATA SET',value=False,key='c')
    if val:
        lefty, righty = sec3.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='3')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-3')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    n_neighbors=sec3.slider('Choose the number of neighbors:', min_value=5, max_value=30, value=5, step=5 ,key='33')
    
    left,mid,right=sec3.columns(3)
    
    weights=left.selectbox('Select the weights function',['uniform','distance'],key='333')
    algorithm=mid.selectbox('Select the algorithm',['auto','ball_tree','kd_tree','brute'],key='3333')
    metric=right.selectbox('Choose the type of distance',['minkowski','euclidean','manhattan'],key='33333')
    
   
    threshold=sec3.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
    
    model=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)

    model.fit(xtrain,ytrain)

    fig_test2=plot(model,xtest,ytest,prob=True, threshold=threshold)
    sec3.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=(model.predict_proba(xtest)[:,1]>=threshold)*1.0
    
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    
    left2,right2=sec3.columns(2)
    
    left2.markdown(strc(acc,prec,recall,f1,'test'))
    sns.set(font_scale = 1.5)
    fig=sns.heatmap(confusion_matrix(ytest,ypreds),annot=True,cmap='viridis',fmt='.0f', annot_kws={'size': 20})
    fig.set(ylabel='True labels', xlabel='Predictions',title='Confusion Matrix')
    right2.pyplot()
    
if selected=='SVM':
    
    sec0,sec1,sec2,sec3 = st.tabs(['Overview','Predictions','Probabilities','Precision-Recall Tradeoff'])
    
    sec0.markdown("""
                  
                  ## Motivation
                  
                  SVMs are incredibly powerful as they have so much flexibility in choosing the kernels and employ advanced optimization
                  processes like the 'kernel trick'. You can choose different kernels and tailor the model to your specific dataset which
                  makes them very useful.
                  
                  Let's see more details about it!
                  
                  ## What is a SVM?""")
    
    img=Image.open('SVM.png')
    sec0.image(img)
    
    sec0.markdown("""
                  - SVMs are powerful algorithms that operate based on a kernel (Function). We can choose different kernels depending on our
                  problem
                  - On the left, we can see a linear kernel being used and on the right, a non-linear kernel is being used
                  - This flexibility makes SVM powerful
                  """)
    sec0.caption("""
                 Image credits:
                - [geeksforgeeks](https://www.geeksforgeeks.org/introduction-to-support-vector-machines-svm/)
                 """)
    
    sec1.markdown("""
    #### Overview: 
    - Increasing the **regularization** parameter prevents overfitting (Fitting to noise)
    - The **kernel** determines the type of function you want to use for classification
    - If you choose the polynomial kernel, you can also select the **degree of the polynomial**
    
    **Try out the CUSTOM DATA SET feature to define your train and test sets!**
                """)
    
    val=sec1.checkbox('USE CUSTOM DATA SET',value=False,key='a')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec1.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='1')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-1')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left,right=sec1.columns(2)
    
    C=left.selectbox('Choose the regularization parameter:',  [1, 100, 10, 0.1, 0.001], key='11')
    kernel=right.selectbox('Select the Kernel',['rbf','poly','linear','sigmoid'],key='111')
    degree=0
    if kernel=='poly':
        degree=sec1.slider('Choose the degree of the polynomial',min_value=1,max_value=10,step=1, value=2,key='1111')
    
    model=SVC(C=C,kernel=kernel,degree=degree)
    model.fit(xtrain,ytrain)
    
    fig_train=plot(model,xtrain,ytrain,train=True)
    sec1.plotly_chart(fig_train,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    

    fig_test=plot(model,xtest,ytest)
    sec1.plotly_chart(fig_test,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    ##################################   Probabilities                 ###########################################################
    
    sec2.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab
                - Adjust different settings to change how likely the model predicts someone might have heart disease
                  """)
    
    val=sec2.checkbox('USE CUSTOM DATA SET',value=False,key='b')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec2.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='2')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-2')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left,right=sec2.columns(2)
    
    C=left.selectbox('Choose the regularization parameter:',  [1, 100, 10, 0.1, 0.001], key='22')
    kernel=right.selectbox('Select the kernel',['rbf','poly','linear','sigmoid'],key='222')
    degree=0
    if kernel=='poly':
        degree=sec2.slider('Choose the degree of the polynomial',min_value=1,max_value=10,step=1,value=2,key='2222')
    
    model=SVC(C=C,kernel=kernel,degree=degree, probability=True)

    model.fit(xtrain,ytrain)
    
    fig_train2=plot(model,xtrain,ytrain,prob=True,train=True)
    sec2.plotly_chart(fig_train2,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    
    fig_test2=plot(model,xtest,ytest,prob=True)
    sec2.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    #############################################    Precision - Recall ##########################################
    
    sec3.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab. 
                - **Normally, models predict 1 if the probability is over 0.5 (default probability threshold). Customize your model by picking 
                the probability threshold!**
                - A high threshold means the model says someone has heart disease **ONLY** if it's 
                absolutely sure!
                - A small threshold makes the model **MORE CAUTIOUS**—it says someone has 
                heart disease even if it's just a little suspicious! **Choose what you want for your problem**
                  """)
    
    val=sec3.checkbox('USE CUSTOM DATA SET',value=False,key='c')
    if val:
        lefty, righty = sec3.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='3')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-3')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left,right=sec3.columns(2)
    
    C=left.selectbox('Choose the regularization parameter:',  [1, 100, 10, 0.1, 0.001], key='33')
    kernel=right.selectbox('Select the kernel',['rbf','poly','linear','sigmoid'],key='333')
    degree=0
    if kernel=='poly':
        degree=sec3.slider('Choose the degree of the polynomial',min_value=1,max_value=10,step=1,value=2,key='3333')
    
   
    threshold=sec3.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
    
    model=SVC(C=C,kernel=kernel,degree=degree, probability=True)

    model.fit(xtrain,ytrain)

    fig_test2=plot(model,xtest,ytest,prob=True, threshold=threshold)
    sec3.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=(model.predict_proba(xtest)[:,1]>=threshold)*1.0
    
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    
    left2,right2=sec3.columns(2)
    
    left2.markdown(strc(acc,prec,recall,f1,'test'))
    sns.set(font_scale = 1.5)
    fig=sns.heatmap(confusion_matrix(ytest,ypreds),annot=True,cmap='viridis',fmt='.0f', annot_kws={'size': 20})
    fig.set(ylabel='True labels', xlabel='Predictions',title='Confusion Matrix')
    right2.pyplot()
    
if selected=='Neural Networks':
    
    overview,sec0,sec1,sec2,sec3 = st.tabs(['Overview','NN Visualization','Predictions','Probabilities','Precision-Recall tradeoff'])
    
    overview.markdown("""
                      
                      ## Motivation
                      
                      Now that you have seen many simple and advanced models, let's dive into one of the most advanced and complex 
                      models - Neural networks. They are incredibly powerful and versatile 
                      as we can customize the structure of the entire network. 
                      
                      **This is the final model that you will work with.
                      In the next section, we will make predictions using the model that you will choose**
                      
                      Let's see more details about Neural Networks!
                      
                      ## What is a Neural Network?""")
    img=Image.open('Neural Networks.png')
    overview.image(img)
    
    overview.markdown("""
                      - Neural networks are powerful, complex models that can create complex decision boundaries
                      - As in the image above, they have an input layer, hidden layer(s) and an output layer
                      - The data passes through different layers and nodes. 
                      - Numerous functions extract patterns from the data and so Neural Networks are very powerful.
                      
                      Look at the image below for similarities between neural networks and the neurons in our brains!
                      """)
    img2 = Image.open('NNcomp.jpg')
    overview.image(img2)
    
    overview.caption("""
                     Image credits:
                      - [javatpoint](https://www.javatpoint.com/artificial-neural-network)
                      - [geeksforgeeks](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/)
                      """)
    
    sec0.markdown("""
    #### Overview: 
    - The way the model is structured depends on its layers and nodes. More layers and nodes make the model more complex.
                """)
    
    layers=sec0.slider('Select the number of hidden layers',min_value=1,max_value=20,value=2,step=1,key='00')
    nodes=sec0.slider('Select the number of nodes per hidden layer', min_value=3, max_value=20,value=5,step=1,key='000')
    nn_config=[len(imp_vars)]+([nodes]*layers)+[1]
    network = create_network(nn_config)
    fig=draw_network(network)
    sec0.pyplot(fig)
    
    sec1.markdown("""
    #### Overview: 
    - The way the model is structured depends on its **layers** and **nodes**. More layers and nodes make the model more complex.
    - The **activation** determines the function we want in each node to classify the points
    - **Optimizer** determines how we arrive at the optimal solution. Increasing the **regularization** term prevents overfitting (Fitting to noise)
    
    **Try out the CUSTOM DATA SET feature to define your train and test sets!**
                """)
    
    val=sec1.checkbox('USE CUSTOM DATA SET',value=False,key='a')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec1.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='1')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-1')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left0,right0=sec1.columns(2)
    
    left,mid,right=sec1.columns(3)
    
    layers=left0.slider('Select the number of hidden layers',min_value=1,max_value=20,value=7,step=1,key='11')
    nodes=right0.slider('Select the number of nodes per hidden layer', min_value=3, max_value=20,value=7,step=1,key='111')
    activation=left.selectbox('Select the activation function', ['relu','tanh','logistic','identity'], key='1111')
    solver=mid.selectbox('Select the optimizer',['adam','sgd','lbfgs'], key='11111')
    alpha=right.selectbox('Select the strength of regularization',[0.0001,0.001,0.01,0.1,1,10],key='10')
    
    
    model=MLPClassifier(hidden_layer_sizes=[nodes]*layers, activation=activation, solver=solver, alpha=alpha)
    model.fit(xtrain,ytrain)
    
    fig_train=plot(model,xtrain,ytrain,train=True)
    sec1.plotly_chart(fig_train,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'train',hor=True))

    fig_test=plot(model,xtest,ytest)
    sec1.plotly_chart(fig_test,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec1.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    ##################################   Probabilities                 ###########################################################
    
    sec2.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab
                - Adjust different settings to change how likely the model predicts someone might have heart disease
                  """)
    
    val=sec2.checkbox('USE CUSTOM DATA SET',value=False,key='b')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec2.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='2')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-2')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left0,right0=sec2.columns(2)
    
    left,mid,right=sec2.columns(3)
    
    layers=left0.slider('Select the number of hidden layers',min_value=1,max_value=20,value=7,step=1,key='22')
    nodes=right0.slider('Select the number of nodes per hidden layer', min_value=3, max_value=20,value=7,step=1,key='222')
    activation=left.selectbox('Select the activation function', ['relu','tanh','logistic','identity'], key='2222')
    solver=mid.selectbox('Select the optimizer',['adam','sgd','lbfgs'], key='22222')
    alpha=right.selectbox('Select the strength of regularization',[0.0001,0.001,0.01,0.1,1,10],key='20')
    
    
    model=MLPClassifier(hidden_layer_sizes=[nodes]*layers, activation=activation, solver=solver, alpha=alpha)
    model.fit(xtrain,ytrain)
    
    fig_train2=plot(model,xtrain,ytrain,prob=True,train=True)
    sec2.plotly_chart(fig_train2,use_container_width = True)
    
    ypreds=model.predict(xtrain)
    acc=accuracy_score(ytrain,ypreds)
    prec=precision_score(ytrain,ypreds)
    recall=recall_score(ytrain,ypreds)
    f1=f1_score(ytrain,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'train',hor=True))
    
    fig_test2=plot(model,xtest,ytest,prob=True)
    sec2.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=model.predict(xtest)
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    sec2.markdown(strc(acc,prec,recall,f1,'test',hor=True))
    
    #############################################    Precision - Recall ##########################################
    
    sec3.markdown("""
                  #### Overview:
                - For description about the parameters, check out the 'Predictions' tab. 
                - **Normally, models predict 1 if the probability is over 0.5 (default probability threshold). Customize your model by picking 
                the probability threshold!**
                - A high threshold means the model says someone has heart disease **ONLY** if it's 
                absolutely sure!
                - A small threshold makes the model **MORE CAUTIOUS**—it says someone has 
                heart disease even if it's just a little suspicious! **Choose what you want for your problem**
                  """)
    
    val=sec3.checkbox('USE CUSTOM DATA SET',value=False,key='c')
    test_size=0.2
    ran=7
    if val:
        lefty, righty = sec3.columns(2)
        test_size=lefty.slider('Choose the test set size:',min_value=0.2,max_value=0.8,step=0.1,key='3')
        ran=righty.selectbox('Select random seed for reproducibility',[7,0,3,10,12,15,20,30,50], key='-3')
        xtrain, xtest, ytrain, ytest = train_test_split(df5[imp_vars],df.HeartDisease,test_size=test_size, random_state=ran)
        xtrain, xtest = reduce(xtrain, xtest)
            
    left0,right0=sec3.columns(2)
    
    left,mid,right=sec3.columns(3)
    
    layers=left0.slider('Select the number of hidden layers',min_value=1,max_value=20,value=7,step=1,key='33')
    nodes=right0.slider('Select the number of nodes per hidden layer', min_value=3, max_value=20,value=7,step=1,key='333')
    activation=left.selectbox('Select the activation function', ['relu','tanh','logistic','identity'], key='3333')
    solver=mid.selectbox('Select the optimizer',['adam','sgd','lbfgs'], key='33333')
    alpha=right.selectbox('Select the strength of regularization',[0.0001,0.001,0.01,0.1,1,10],key='30')
    
   
    threshold=sec3.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
    
    model=MLPClassifier(hidden_layer_sizes=[nodes]*layers, activation=activation, solver=solver, alpha=alpha)
    
    model.fit(xtrain,ytrain)

    fig_test2=plot(model,xtest,ytest,prob=True, threshold=threshold)
    sec3.plotly_chart(fig_test2,use_container_width = True)
    
    ypreds=(model.predict_proba(xtest)[:,1]>=threshold)*1.0
    
    acc=accuracy_score(ytest,ypreds)
    prec=precision_score(ytest,ypreds)
    recall=recall_score(ytest,ypreds)
    f1=f1_score(ytest,ypreds)
    
    left2,right2=sec3.columns(2)
    
    left2.markdown(strc(acc,prec,recall,f1,'test'))
    sns.set(font_scale = 1.5)
    fig2,ax=plt.subplots()
    ax=sns.heatmap(confusion_matrix(ytest,ypreds),annot=True,cmap='viridis',fmt='.0f', annot_kws={'size': 20})
    ax.set(ylabel='True labels', xlabel='Predictions',title='Confusion Matrix')
    right2.pyplot(fig2)
    
if selected=='Summary':
    gender_map = {'Male':'M','Female':'F'}
    cp_map = {'Atypical Angina':'ATA','Non-anginal Pain':'NAP','Asymptomatic':'ASY','Typical Angina':'TA'}
    angina_map = {'Yes':'Y','No':'N'}
    fbs_map = {'Yes':1.0, 'No':0.0}
    
    
    sec1, sec2 = st.tabs(['Make predictions!','Conclusion and Future works'])
    
    sec1.markdown("""
                  ## Overview
                  
                  Now that you have seen all the models and have an idea about what model you want to use for your problem, choose a model 
                  below and let's make predictions!
                  
                  **Note:** You can use the default model or alter the model parameters **BY SELECTING THE CUSTOM MODEL OPTION**
                  """)
    sec1.write("## Step 1: Select the Model you would like")
    m=sec1.selectbox('Choose the model you would like',['Logistic Regression','Decision Tree','Random Forest','KNN Classifier','SVM'
                                                      , 'Neural Networks'])
    
    model_map={'Logistic Regression': LogisticRegression,'Decision Tree': tree.DecisionTreeClassifier, 'Random Forest':RandomForestClassifier, 
               'KNN Classifier':KNeighborsClassifier, 'SVM':SVC, 'Neural Networks': MLPClassifier}
    model=model_map[m]()
    
    sel = sec1.radio('Select the type of model',['Default Model','Custom Model'],index=0,key='a')
    
    if sel=='Custom Model':
        if m=='Logistic Regression':
            left4,mid4,right4=sec1.columns(3)
            solver=left4.selectbox('Choose solver:', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],key='33')
            if solver=='newton-cg' or solver=='lbfgs' or solver=='sag':
                pen=['l2','none']
            elif solver=='liblinear':
                pen=['l1','l2']
            else:
                pen=['none','l1','l2','elasticnet'] 
            penalty=mid4.selectbox('Choose the penalty',pen,key='333')
            if penalty=='none':
                penalty=None
            C=right4.selectbox('Choose penalty strength',[100,10,1,0.1,0.01],key='3333')
            threshold=sec1.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='33333')
            model=LogisticRegression(solver=solver,penalty=penalty,C=C)
        elif m=='Decision Tree':
            left4,right4=sec1.columns(2)
           
            min_samples_split=left4.selectbox('Choose the minimum samples to split a node',[2,5,10,15,20,30,50],key='333')
            min_samples_leaves=right4.selectbox('Choose the minimum samples in a leaf node',[1,5,10,15,20,30,50],key='3333')
           
            threshold=sec1.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='33333')
            
            model=tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves)
        elif m=='Random Forest':
            max_depth=sec1.slider('Choose the height of the tree:', min_value=5, max_value=30, value=30 ,key='33')
            
            left4,mid4,right4=sec1.columns(3)
            
            min_samples_split=left4.selectbox('Min samples to split a node',[2,5,10,15,20,30,50],key='333')
            min_samples_leaves=mid4.selectbox('Min samples in a leaf node',[1,5,10,15,20,30,50],key='3333')
            n_estimators=right4.selectbox('Choose the number of estimators',[201,101,51,31,11,1],key='33333')
            
           
            threshold=sec1.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
            
            model=RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaves,
                                         n_estimators=n_estimators)
        elif m=='KNN Classifier':
            n_neighbors=sec1.slider('Choose the number of neighbors:', min_value=5, max_value=30, value=5, step=5 ,key='33')
            
            left4,mid4,right4=sec1.columns(3)
            
            weights=left4.selectbox('Select the weights function',['uniform','distance'],key='333')
            algorithm=mid4.selectbox('Select the algorithm',['auto','ball_tree','kd_tree','brute'],key='3333')
            metric=right4.selectbox('Choose the type of distance',['minkowski','euclidean','manhattan'],key='33333')
            
           
            threshold=sec1.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
            
            model=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)
        elif m=='SVM':
            left4,right4=sec1.columns(2)
            
            C=left4.selectbox('Choose the regularization parameter:',  [1, 100, 10, 0.1, 0.001], key='33')
            kernel=right4.selectbox('Select the kernel',['rbf','poly','linear','sigmoid'],key='333')
            degree=0
            if kernel=='poly':
                degree=sec1.slider('Choose the degree of the polynomial',min_value=1,max_value=10,step=1,value=2,key='3333')
            
           
            threshold=sec1.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
            
            model=SVC(C=C,kernel=kernel,degree=degree, probability=True)
        elif m=='Neural Networks':
            left4,right4=sec1.columns(2)
            
            left5,mid5,right5=sec1.columns(3)
            
            layers=left4.slider('Select the number of hidden layers',min_value=1,max_value=20,value=7,step=1,key='33')
            nodes=right4.slider('Select the number of nodes per hidden layer', min_value=3, max_value=20,value=7,step=1,key='333')
            activation=left5.selectbox('Select the activation function', ['relu','tanh','logistic','identity'], key='3333')
            solver=mid5.selectbox('Select the optimizer',['adam','sgd','lbfgs'], key='33333')
            alpha=right5.selectbox('Select the strength of regularization',[0.0001,0.001,0.01,0.1,1,10],key='30')
            
           
            threshold=sec1.slider('Choose probability threshold',min_value=0.1,max_value=0.9,step=0.1,value=0.5,key='333333')
            
            model=MLPClassifier(hidden_layer_sizes=[nodes]*layers, activation=activation, solver=solver, alpha=alpha)

    model.fit(res, df5.HeartDisease)
    
    custom = {'Age':[],'MaxHR':[],'Oldpeak':[],'Sex':[],'ChestPainType':[],'FastingBS':[],'ExerciseAngina':[],'ST_Slope':[]}
    
    sec1.write("## Step 2: Input the data for which you want to predict:")
    
    left, mid, right = sec1.columns(3)
    left2, mid2, right2 = sec1.columns(3)
    left3,right3 = sec1.columns(2)
    
    sex=left.selectbox('Gender',['Male', 'Female'])
    cp=mid.selectbox('Chest Pain Type',['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'])
    fbs=right.selectbox('Fasting Blood Sugar',['Yes','No'])
    ea = left2.selectbox('Exercise Angina',['Yes','No'])
    st=mid2.selectbox('ST Slope',['Up','Flat','Down'])
    op=right2.slider('Oldpeak',min_value=df5.Oldpeak.min(),max_value=df5.Oldpeak.max())
    age=left3.slider('Age',min_value=25,max_value=100)
    maxhr=right3.slider('Max Heart Rate', min_value=50, max_value=210)
    
    custom['Age'].append(age)
    custom['MaxHR'].append(maxhr)
    custom['Oldpeak'].append(op)
    custom['Sex'].append(gender_map[sex])
    custom['ChestPainType'].append(cp_map[cp])
    custom['FastingBS'].append(fbs_map[fbs])
    custom['ExerciseAngina'].append(angina_map[ea])
    custom['ST_Slope'].append(st)
    
    custom_df = pd.DataFrame(custom)
    _,custom_df=reduce(df5[imp_vars],pd.concat([df5[imp_vars].tail(100),custom_df]))
    custom_df=custom_df.tail(1)
    
    if sel=='Custom Model':
        pred=np.array([model.predict_proba(custom_df)[:,1]])
        pred=(pred>=threshold)*1
        fig=plot(model, custom_df, np.reshape(pred,(1,)), prob= True, threshold = threshold, train=False, Final=True)
    else:
        pred=np.array([model.predict(custom_df)])
        fig=plot(model, custom_df, np.reshape(pred,(1,)), prob= False, threshold = None, train=False, Final=True)
        
    fig.update_layout(legend=dict(yanchor="top", xanchor="left", x=-0.3, y=0.975, title='Custom Data Point'),
                      legend2=dict(title='Prediction Area', y=0.975, title_font_size=12.75, x=1.05), 
                      title_text='Custom Prediction', title_x=0.45)
    sec1.plotly_chart(fig)
    
    if pred[0][0]==0:
        sec1.write('#### :green[A person with these attributes is predicted NOT to have heart disease!]')
    elif pred[0][0]==1:
        sec1.write('#### :red[A person with these attributes is predicted to have heart disease!]')
        
    
    sec2.markdown("""
                  ## Conclusion and Future works
                  """)
    img=Image.open('conclusion.jpg')
    sec2.image(img)
    
    sec2.markdown("""
                  - We have reduced the dataset to 2 dimensions to ease computation and visualize powerful algorithms
                  - We have deep-dived into multiple models and made predictions with models tailor-made for the dataset and our purposes
                  - **What next?** The next step in the future would be to deploy the model to a mobile app and make live predictions based on biometrics
                  collected from the user!!
                  """)
                  
    sec2.caption("""
                 Image credits:
                - [elgl](https://elgl.org/app-for-that-native-mobile-is-a-civic-tech-foregone-conclusion/)
                 """)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


