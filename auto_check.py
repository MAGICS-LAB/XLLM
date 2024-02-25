import pandas as pd





data = pd.read_csv('dataset/harmful_response.csv')

list1 = []
for i  in range(len(data)):
  
    if 'As a responsible' in data['text'][i]:
      list1.append(i)
      
      
print(list1)