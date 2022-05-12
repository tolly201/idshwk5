import pandas
import json
import math
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
import  pickle
import warnings
warnings.filterwarnings(action='ignore')
scaler = preprocessing.StandardScaler()
vectorizer = DictVectorizer(sparse=False)
train_pandas = pandas.DataFrame(columns=["domain","label","root","length_of_domain","root_class_knowledge","root_class_no_knowledge","entropy","segmentation","number_rate","vowel_rate","result"])
test_pandas = pandas.DataFrame(columns=["domain","root","length_of_domain","root_class_knowledge","root_class_no_knowledge","entropy","segmentation","number_rate","vowel_rate"])
train_domain_set={}
train_label_set={}
test_domain_set={}
root_famous=['cn','com','cc','net','org','gov','info']
root_all =list()
counter_of_root=0
train_file="train.txt"
test_file = "test.txt"
counter = 0
help_str= 'a'
help_str1 = 'a'
max_of_train = 10000
max_of_test = 10000
def getentropy(domain):
    tmp_dict = {}
    domain_len = len(domain)
    for i in range(0, domain_len):
        if domain[i] in tmp_dict.keys():
            tmp_dict[domain[i]] = tmp_dict[domain[i]] + 1
        else:
            tmp_dict[domain[i]] = 1
    shannon = float(0)
    for i in tmp_dict.keys():
        p = float(tmp_dict[i]) / domain_len
        shannon = shannon - p * math.log(p, 2)
    return shannon
def getnumbers(domain):
    number_set={'1','2','3','4','5','6','7','8','9','0'}
    counter_of_number=0
    result = float(0)
    for i in range(0,len(domain)):
        if domain[i] in number_set:
            counter_of_number+=1
    result = counter_of_number/len(domain)
    return result
def getvowel(domain):
    number_set={'a','e','i','o','u','A','E','I','O','U'}
    counter_of_vowel=0
    result = float(0)
    for i in range(0,len(domain)):
        if domain[i] in number_set:
            counter_of_vowel+=1
    result = counter_of_vowel/len(domain)
    return result
max_of_train = input("Please enter the number of data for training\n")
max_of_test = input("Please enter the number of data for test\n")
max_of_test=int(max_of_test)
max_of_train = int(max_of_train)
train_file_flow = open("train.txt")
print("file opened, start read train data\n")
for line in train_file_flow:
    help_str = line.split(",")
#    train_domain_set[counter]=help_str[0]

    help_str1 = help_str[1].split("\n")
#    train_label_set[counter]=help_str1[0]
    train_pandas=train_pandas.append({'domain' : help_str[0],'label' : help_str1[0],'length_of_domain' : help_str[0].__len__() }, ignore_index=True)
    if help_str1[0]=='dga':
        train_pandas.loc[counter,'result']=0
    if help_str1[0]=='notdga':
        train_pandas.loc[counter,'result']=1
    train_pandas.loc[counter,'entropy']=getentropy(help_str[0])
    train_pandas.loc[counter,'number_rate']=getnumbers(help_str[0])
    train_pandas.loc[counter,'vowel_rate']=getvowel(help_str[0])
    help_str =help_str[0].split(".")
    train_pandas.loc[counter,'segmentation']=help_str.__len__()
    train_pandas.loc[counter,'root']=help_str[-1]
    if help_str[-1] in root_famous:
        train_pandas.loc[counter, 'root_class_knowledge'] = 1
    else:
        train_pandas.loc[counter, 'root_class_knowledge'] = 0
    if help_str[-1] in root_all:
        train_pandas.loc[counter, 'root_class_no_knowledge'] = root_all.index(help_str[-1])
    else:
        root_all.append(help_str[-1])
        train_pandas.loc[counter, 'root_class_no_knowledge'] = root_all.index(help_str[-1])
#        counter_of_root+=1
    counter+=1
    if counter >= max_of_train:
        break
print("finish read train data\n")
print("start read test data\n")
train_pandas.to_csv("train.csv",index=False)
counter = 0
train_file_flow_test = open("test.txt")
for line in train_file_flow_test:
    help_str = line.split("\n")
    test_pandas=test_pandas.append({'domain' : help_str[0],'length_of_domain' : help_str[0].__len__() }, ignore_index=True)
    test_pandas.loc[counter,'entropy']=getentropy(help_str[0])
    test_pandas.loc[counter,'number_rate']=getnumbers(help_str[0])
    test_pandas.loc[counter,'vowel_rate']=getvowel(help_str[0])
    help_str =help_str[0].split(".")
    test_pandas.loc[counter,'segmentation']=help_str.__len__()
    test_pandas.loc[counter,'root']=help_str[-1]
    if help_str[-1] in root_famous:
        test_pandas.loc[counter, 'root_class_knowledge'] = 1
    else:
        test_pandas.loc[counter, 'root_class_knowledge'] = 0
    if help_str[-1] in root_all:
        test_pandas.loc[counter, 'root_class_no_knowledge'] = root_all.index(help_str[-1])
    else:
        root_all.append(help_str[-1])
        test_pandas.loc[counter, 'root_class_no_knowledge'] = root_all.index(help_str[-1])
#        counter_of_root+=1
    counter+=1
    if counter >= max_of_test:
        break
print("finish read test data")
print("solving data")

test_pandas.to_csv("test.csv",index=False)
train_data = train_pandas[["length_of_domain","root_class_knowledge","root_class_no_knowledge","entropy","segmentation","number_rate","vowel_rate"]]
test_data = test_pandas[["length_of_domain","root_class_knowledge","root_class_no_knowledge","entropy","segmentation","number_rate","vowel_rate"]]
X = train_data.values[:, 0:7]
Y = train_pandas.values[:,-1]
print("start training")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y.astype('int'))
print("stop training")
print("save model")
with open('clf.pickle','wb') as f:
    pickle.dump(clf,f)
result = clf.predict(test_data.values[:,0:7])

print("solve result")
file = open('result.txt','w')
for i in range(0,(len(result))):
    file.write(test_pandas.loc[i,'domain']+',');
    if result[i]==1:
        file.write('notdga\n')
    if result[i]==0:
        file.write('dga\n')
file.close()
print("finish")