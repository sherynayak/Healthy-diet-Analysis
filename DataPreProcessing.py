import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D:\\ICT\\MainProject\\VSCode\\Healthy_Diet_Data.csv')

data.dtypes
data.shape

data.columns

data['Undernourished'] = data['Undernourished'].str.replace('<2.5','2.5')
data['Undernourished'] = pd.to_numeric(data['Undernourished'])

data = data.drop(['Unit (all except Population)'],axis =1)

data.columns = data.columns.str.replace('&', 'And')
data.columns = data.columns.str.replace('- ', '')
data.columns = data.columns.str.replace(',', '')
data.columns = data.columns.str.replace(' ', '_')


data.isnull().sum().sort_values(ascending=False)

data.describe()

data.info()

data.corr()

sns.heatmap(data.corr())
sns.heatmap(data.corr(), annot=True)
plt.title('Correlations of Healthy Diet')

Obesity_Null = data[data['Obesity'].isnull()]
Undernourished_Null = data[data['Undernourished'].isnull()]
Confirmed_Null = data[data['Confirmed'].isnull()]
Deaths_Null = data[data['Deaths'].isnull()]
Recovered_Null = data[data['Recovered'].isnull()]
Active_Null = data[data['Active'].isnull()]

Confirmed_Null_N = Confirmed_Null[['Country','Confirmed','Deaths','Recovered','Active']]

#Active_Null.to_csv(r'D:\ICT\MainProject\CARD_Null.csv', index = False, header=True)
#sns.heatmap(data.corr(), annot=True)
#data.corr()

# grains = data.iloc[:,[0,5,13,14,15,16,17,18,19,20]]
# vegetables = data.iloc[:,[0,21,22,23]]
# fruits = data.iloc[:,[0,1,8]]
# protein = data.iloc[:,[0,2,3,4,6,7,9,10,11,12]]

data['Grains'] = data.iloc[:,[5,13,14,15,16,17,18,19,20]].sum(axis=1)
data['Vegetables_Sum'] = data.iloc[:,[21,22,23]].sum(axis=1)
data['Fruits'] = data.iloc[:,[1,8]].sum(axis=1)
data['Protein'] = data.iloc[:,[2,3,4,6,7,9,10,11,12]].sum(axis=1)
#sample = data.iloc[:,11]
data['Continent'] = ""
south_america_countries = {'Brazil', 'Colombia', 'Argentina',
                         'Peru', 'Venezuela', 'Chile', 'Ecuador', 
                         'Bolivia', 'Paraguay', 'Uruguay', 'Guyana', 
                         'Suriname', 'French Guiana', 'Falkland Islands'}
                        
for i in south_america_countries:
    data.loc[(data['Country'] == i), 'Continent'] = 'South America' 




europian_countries = {'Russia', 'Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 
                            'Ukraine', 'Poland', 'Romania', 'Netherlands', 'Belgium', 
                            'Czechia', 'Czechia', 'Greece', 'Portugal', 'Sweden', 
                            'Hungary', 'Belarus', 'Austria', 'Serbia', 'Switzerland', 
                            'Bulgaria', 'Denmark', 'Finland', 'Slovakia', 'Norway', 
                            'Ireland', 'Croatia', 'Moldova', 'Bosnia and Herzegovina', 
                            'Albania', 'Lithuania', 'North Macedonia', 'Slovenia' , 
                            'Latvia', 'Estonia', 'Montenegro',  'Luxembourg', 
                            'Malta', 'Iceland', 'Channel Islands', 'Isle of Man' , 
                            'Andorra', 'Faeroe Islands', 'Monaco', 'Liechtenstein', 
                            'San Marino', 'Gibraltar', 'Holy See'}
                        
for i in europian_countries:
    data.loc[(data['Country'] == i), 'Continent'] = 'Europe' 

african_countries = {'Nigeria','Ethiopia', 'Egypt', 'Democratic Republic of the Congo', 
                    'Tanzania', 'South Africa', 'Kenya', 'Uganda', 'Algeria', 'Sudan', 
                    'Morocco', 'Angola', 'Mozambique', 'Ghana', 'Madagascar', 'Cameroon', 
                    'Niger', 'Burkina Faso', 'Mali', 'Malawi', 'Zambia', 'Senegal', 
                    'Chad', 'Somalia', 'Zimbabwe', 'Guinea', 'Rwanda', 'Benin', 'Burundi', 
                    'Tunisia', 'South Sudan', 'Togo', 'Sierra Leone', 'Libya', 'Congo', 
                    'Liberia', 'Central African Republic', 'Mauritania', 'Eritrea', 
                    'Namibia', 'Gambia', 'Botswana', 'Gabon', 'Lesotho', 'Guinea-Bissau', 
                    'Equatorial Guinea', 'Mauritius', 'Eswatini', 'Djibouti', 'Réunion', 
                    'Comoros', 'Western Sahara', 'Cabo Verde', 'Mayotte', 
                    'Sao Tome and Principe', 'Seychelles', 'Saint Helena'}
                        
for i in african_countries:
    data.loc[(data['Country'] == i), 'Continent'] = 'Africa' 

australian_countries = {'Australia', 'Papua New Guinea', 'New Zealand', 'Fiji', 
                        'Solomon Islands', 'Vanuatu', 'New Caledonia', 'French Polynesia', 
                        'Samoa', 'Guam', 'Kiribati', 'Micronesia', 'Tonga', 
                        'Marshall Islands', 'Northern Mariana Islands', 'American Samoa', 
                        'Palau', 'Cook Islands', 'Tuvalu', 'Wallis and Futuna Islands', 
                        'Nauru', 'Niue', 'Tokelau'}
                        
for i in australian_countries:
    data.loc[(data['Country'] == i), 'Continent'] = 'Australia' 

north_america_countries = {'U.S.A', 'Mexico' ,'Canada', 'Guatemala' ,'Haiti', 'Cuba', 
                        'Dominican Republic', 'Honduras', 'Nicaragua', 'El Salvador', 
                        'Costa Rica', 'Panama', 'Jamaica', 'Puerto Rico', 
                        'Trinidad and Tobago', 'Guadeloupe', 'Belize', 'Bahamas', 
                        'Martinique', 'Barbados', 'Saint Lucia', 'Curaçao', 'Grenada', 
                        'Saint Vincent and the Grenadines', 'Aruba', 
                        'United States Virgin Islands', 'Antigua and Barbuda', 
                        'Dominica', 'Cayman Islands', 'Bermuda', 'Greenland', 
                        'Saint Kitts and Nevis', 'Sint Maarte', 'Turks and Caicos Islands', 
                        'Saint Martin', 'British Virgin Islands', 'Caribbean Netherlands' ,
                        'Anguilla', 'Saint Barthélemy', 'Saint Pierre and Miquelon', 
                        'Montserrat'}
                        
for i in north_america_countries:
    data.loc[(data['Country'] == i), 'Continent'] = 'North America' 


asian_countries = {'China', 'India','Indonesia','Pakistan','Bangladesh','Japan','Philippines','Vietnam','Turkey','Iran','Thailand','Myanmar','South Korea','Iraq','Afghanistan','Saudi Arabia','Uzbekistan','Malaysia','Yemen','Nepal','North Korea','Taiwan','Sri Lanka','Kazakhstan','Syria','Cambodia','Jordan','Azerbaijan','United Arab Emirates','Tajikistan','Israel','Hong Kong','Laos','Lebanon','Kyrgyzstan','Turkmenistan','Singapore','Oman','State of Palestine','Kuwait','Georgia','Mongolia','Armenia','Qatar','Bahrain','Timor-Leste','Cyprus','Bhutan','Macao','Maldives','Brunei Darussalam'}   

for i in asian_countries:
    data.loc[(data['Country'] == i), 'Continent'] = 'Asia' 

ContinentNull = data[data['Continent'] == ""]

data.loc[(data['Country'] == "Cote d'Ivoire"), 'Continent'] = 'Africa'
data.loc[(data['Country'] == "Iran (Islamic Republic of)"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "Korea, North"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "Korea, South"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "Lao People's Democratic Republic"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "Republic of Moldova"), 'Continent'] = 'Europe'
data.loc[(data['Country'] == "Russian Federation"), 'Continent'] = 'Europe'
data.loc[(data['Country'] == "Saudi Arabia"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "Taiwan*"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "United Arab Emirates"), 'Continent'] = 'Asia'
data.loc[(data['Country'] == "United Republic of Tanzania"), 'Continent'] = 'Africa'
data.loc[(data['Country'] == "United States of America"), 'Continent'] = 'North America'
data.loc[(data['Country'] == "Venezuela (Bolivarian Republic of)"), 'Continent'] = 'South America'


#sns.barplot(x = data['Country'], y=data['Obesity'])
Active_Null = data[data['Active'].isnull()]
active_null_country = Active_Null[['Country','Continent']]

Obesity_Null = data[data['Obesity'].isnull()]
Obesity_null_country = Obesity_Null[['Country','Continent','Grains','Vegetables_Sum','Fruits','Protein']]
Obesity_null_country['Total'] = Obesity_Null[['Grains','Vegetables_Sum','Fruits','Protein']].sum(axis = 1)

Obesity_null_country['Grains_25'] = Obesity_null_country['Grains']/30*25
Obesity_null_country['Vegetables_Sum_25'] = Obesity_null_country['Vegetables_Sum']/40*25
Obesity_null_country['Fruits_25'] = Obesity_null_country['Fruits']/10*25
Obesity_null_country['Protein_25'] = Obesity_null_country['Protein']/20*25

Obesity_null_country['Max_25'] = Obesity_null_country[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1)

#df.loc[df['column name'] condition, 'new column name'] = 'value if condition is met'
Obesity_null_country.loc[Obesity_null_country['Protein_25'] == Obesity_null_country[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1), 'Max_of_4'] = 'p'

data['Grains_25'] = data['Grains']/30*25
data['Vegetables_Sum_25'] = data['Vegetables_Sum']/40*25
data['Fruits_25'] = data['Fruits']/10*25
data['Protein_25'] = data['Protein']/20*25

data['Total_25'] = data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].sum(axis = 1)

data.loc[data['Protein_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1), 'Max_of_4'] = 'p'
data.loc[data['Vegetables_Sum_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1), 'Max_of_4'] = 'v'
data.loc[data['Fruits_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1), 'Max_of_4'] = 'f'
data.loc[data['Grains_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1), 'Max_of_4'] = 'g'

data.loc[data['Protein_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1), 'Min_of_4'] = 'p'
data.loc[data['Vegetables_Sum_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1), 'Min_of_4'] = 'v'
data.loc[data['Fruits_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1), 'Min_of_4'] = 'f'
data.loc[data['Grains_25'] == data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1), 'Min_of_4'] = 'g'

grains = data[data['Max_of_4'] == 'g']
vegetables = data[data['Max_of_4'] == 'v']
fruits = data[data['Max_of_4'] == 'f']
protein = data[data['Max_of_4'] == 'p']

Obesity_null = data[data['Obesity'].isnull()]
Obesity_null_N = Obesity_null[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25','Obesity','Max_of_4']]
Undernourished_null = data[data['Undernourished'].isnull()]
Undernourished_null_N = Undernourished_null[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25','Undernourished','Min_of_4']]

protein_mean = data[data['Max_of_4'] == 'p'].mean()
grains_mean = data[data['Min_of_4'] == 'g'].mean()

#data.Elimination_Week_Number = data.Elimination_Week_Number.fillna(data.Elimination_Week_Number.mean())
data['Obesity'] = data['Obesity'].fillna(data['Obesity'][data['Max_of_4'] == 'p'].mean())

data['Undernourished'] = data['Undernourished'].fillna(data['Undernourished'][data['Min_of_4'] == 'g'].mean())

data = data.dropna()


# data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1)>130, 'Healthy_Or_Not'] = 'N'
# data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1)<70, 'Healthy_Or_Not'] = 'N'
# data['Healthy_Or_Not'] = data['Healthy_Or_Not'].fillna('H')

data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1)>35, 'Healthy_Or_Not'] = 0
data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1)<15, 'Healthy_Or_Not'] = 0
data['Healthy_Or_Not'] = data['Healthy_Or_Not'].fillna(1)

data['Balance']= data.loc[(data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1)>35)-35]
data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1)<15, 'Balance']-25

# data = data.drop(['Healthy_Or_Not'], axis = 1)

# data['Confirmed'] = data['Confirmed'].fillna(0)
# data['Deaths'] = data['Deaths'].fillna(0)
# data['Recovered'] = data['Recovered'].fillna(0)
# data['Active'] = data['Active'].fillna(0)

data['Deaths_rate'] = (data['Deaths']/data['Confirmed'])*100
data['Recovered_rate'] = (data['Recovered']/data['Confirmed'])*100
data['Active_rate'] = (data['Active']/data['Confirmed'])*100

# data['Deaths_rate'] = data['Deaths_rate'].fillna(0)
# data['Recovered_rate'] = data['Recovered_rate'].fillna(0)
# data['Active_rate'] = data['Active_rate'].fillna(0)

data['Total_rate'] = data[['Deaths_rate','Recovered_rate','Active_rate']].sum(axis = 1)




first_cols = ['Country','Continent']
last_cols = [col for col in data.columns if col not in first_cols]
data = data[first_cols+last_cols]

# fruits_25_above_40 = data[data['Fruits_25']>40]
# fruits_25_above_40 = fruits_25_above_40[['Alcoholic_Beverages','Fruits_Excluding_Wine','Fruits','Fruits_25']]


recover_G_50 = data[(data['Recovered_rate'] >= 50)]
recover_L_50 = data[(data['Recovered_rate'] <= 50)]

#NotHealthy = data['Country'][data['Healthy_Or_Not']==0]

data_describe = data.describe()
data.isnull().sum().sort_values(ascending=False)

data.to_csv(r'D:\ICT\MainProject\MainProject.csv', index = False, header=True)


val = data['Healthy_Or_Not'][data['Country'] == 'India']
print('Value = ', val.values)
if val.values == 0:
    print('Not Healthy')
else:
    print('Healthy')

X = data.iloc[:,np.r_[36:40]]
y = data['Healthy_Or_Not']

# X = data.iloc[:,36:47]
# X = X.drop(['Max_of_4','Min_of_4','Recovered_rate'],axis=1)

# y = y.to_frame()
# healthy_Or_Not = {'N':0, 'H':1}
# y['Healthy_Or_Not'] = [healthy_Or_Not[item] for item in y['Healthy_Or_Not']]


#X = data.iloc[:,np.r_[1:34,42,44]]
# X = data.iloc[:,np.r_[36:39,42,44]]
# y = data['Recovered_rate']

#X = data.drop(['Country','Recovered'],axis=1)

#X.corr()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

# from sklearn.linear_model import LinearRegression
# lm = LinearRegression()
# lm.fit(X_train,y_train)
# y_pred = lm.predict(X_test)

# lm.score(X_test,y_test)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25 , criterion= 'entropy',max_depth= 8, max_features='sqrt', random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('confusion_matrix is : \n', cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print('accuracy_score : ', ac)

from sklearn.metrics import classification_report
cScr = classification_report(y_test,y_pred)
print('classification_report : \n', cScr)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=1)
classifier.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('confusion_matrix is : \n', cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print('accuracy_score : ', ac)

from sklearn.metrics import classification_report
cScr = classification_report(y_test,y_pred)
print('classification_report : \n', cScr)


#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Metrics: \n",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy_score: ",ac)

from sklearn.metrics import classification_report
cScr = classification_report(y_test,y_pred)
print("classification_report: \n",cScr)


#SVC
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Metrics: \n",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy_score: ",ac)

from sklearn.metrics import classification_report
cScr = classification_report(y_test,y_pred)
print("classification_report: \n",cScr)


#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=1)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Metrics: \n",cm)
sns.heatmap(cm,annot=True)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy_score: ",ac)

from sklearn.metrics import classification_report
cScr = classification_report(y_test,y_pred)
print("classification_report: \n",cScr)

# y_test.to_csv(r'D:\ICT\MainProject\y_test.csv', index = False, header=True)
# output = X_test['Deaths_rate']
# output = output.to_frame()
# output['Recovered_rate'] = y_pred
# output = output.drop(['Deaths_rate'],axis=1)
# output.to_csv(r'D:\ICT\MainProject\y_pred.csv', index = False, header=True)