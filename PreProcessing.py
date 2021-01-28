import pandas as pd
import numpy as np

data = pd.read_csv('D:\\ICT\\MainProject\\VSCode\\Healthy_Diet_Data.csv')

data['Undernourished'] = data['Undernourished'].str.replace('<2.5','2.5')
data['Undernourished'] = pd.to_numeric(data['Undernourished'])

data = data.drop(['Unit (all except Population)'],axis =1)

data.columns = data.columns.str.replace('&', 'And')
data.columns = data.columns.str.replace('- ', '')
data.columns = data.columns.str.replace(',', '')
data.columns = data.columns.str.replace(' ', '_')

data['Grains'] = data.iloc[:,[5,13,14,15,16,17,18,19,20]].sum(axis=1)
data['Vegetables_Sum'] = data.iloc[:,[21,22,23]].sum(axis=1)
data['Fruits'] = data.iloc[:,[1,8]].sum(axis=1)
data['Protein'] = data.iloc[:,[2,3,4,6,7,9,10,11,12]].sum(axis=1)

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

data['Obesity'] = data['Obesity'].fillna(data['Obesity'][data['Max_of_4'] == 'p'].mean())
data['Undernourished'] = data['Undernourished'].fillna(data['Undernourished'][data['Min_of_4'] == 'g'].mean())
data = data.dropna()

data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].max(axis = 1)>35, 'Healthy_Or_Not'] = 0
data.loc[data[['Grains_25','Vegetables_Sum_25','Fruits_25','Protein_25']].min(axis = 1)<15, 'Healthy_Or_Not'] = 0
data['Healthy_Or_Not'] = data['Healthy_Or_Not'].fillna(1)

data['Deaths_rate'] = (data['Deaths']/data['Confirmed'])*100
data['Recovered_rate'] = (data['Recovered']/data['Confirmed'])*100
data['Active_rate'] = (data['Active']/data['Confirmed'])*100

data['Total_rate'] = data[['Deaths_rate','Recovered_rate','Active_rate']].sum(axis = 1)

first_cols = ['Country','Continent']
last_cols = [col for col in data.columns if col not in first_cols]
data = data[first_cols+last_cols]

X = data.iloc[:,np.r_[36:40]]
y = data['Healthy_Or_Not']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25 , criterion= 'entropy',max_depth= 8, max_features='sqrt', random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('confusion_matrix is : ', cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print('accuracy_score : ', ac)

# 30% grains, 40% vegetables, 10% fruits, and 20% protein
def healthy_diet(grains, vegetables, fruits, protein):
    grains_diet = (grains/30)*25
    vegetables_diet = (vegetables/40)*25
    fruits_diet = (fruits/10)*25
    protein_diet = (protein/20)*25

    output = classifier.predict([[grains_diet,vegetables_diet,fruits_diet,protein_diet]])

    if(output == 1):
        print('Healthy')
        return 'Following a Healthy'
    else:
        print('Not Healthy')
        return 'Not Following a Healthy'
