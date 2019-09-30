import pandas as pd
import DecisionTreeTools

# import the dataset as a pandas dataframe

df = pd.read_csv("titanic.csv")
df.head()

#dropping the unnecessary attributes
inputs = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Survived'],axis='columns')

#setting the target attribute
target = df['Survived']

#importing sklearn library
from sklearn.preprocessing import LabelEncoder

#labelling the non integer attributes
le_Sex = LabelEncoder()
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])

#dropping the non integer atrributes
inputs_n = inputs.drop(['Sex'],axis='columns')

#filling missing values to '0'
inputs_n = inputs_n.fillna(0) 

inputs_n['Survived'] = target

# preprocess to add tag for continuous vs nominal
parsed_df = inputs_n

# build decision tree
tree_root = DecisionTreeTools.build_decision_tree(parsed_df, 25, 25)

# print out the tree
DecisionTreeTools.print_tree(tree_root, 0)

#testing the above tree
example = parsed_df.iloc[0]

def classify_example(example, tree_root):
    question =  list(tree_root.items())[0][1]
    value = list(tree_root.items())[1][1]

    # ask question
    if example[question] <= float(value):
        answer = list(list(tree_root.items())[2][1].items())[0][1]
    else:
        answer = list(list(tree_root.items())[2][1].items())[1][1]
    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)
val = classify_example(example, tree_root)
if (target[val] == 0):
    print("The passenger died!")
else:
    print("The passenger survived")
