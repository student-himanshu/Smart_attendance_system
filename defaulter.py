import pandas as pd
data=pd.read_csv('record.csv')
x=data.groupby('Name').count()['time']<16
defaulter=x[x].index
df1 = pd.DataFrame(data=defaulter,columns=['Name'])
df1.to_csv('defaulter.csv',index=False)
# print ('df1')
