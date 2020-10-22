import pandas as pd

df = pd.read_csv('games.csv')
pd.set_option('display.max_columns', None)

df.head(2)

df_moves = pd.DataFrame(df.moves.str.split(' ').tolist(),index=df.id)

df_moves_reset = df_moves.reset_index([0,'id'])

df = pd.merge(df,df_moves_reset, on='id')

del df['moves']

df = df.drop(df.columns[35:],axis=1)

df.rename(columns={0:'firstplay',1:'secondplay',2:'thirdplay',3:'fourthplay',4:'fifthplay',5:'sixthplay',
                   6:'seventhplay',7:'eighthplay',8:'ninthplay',9:'tenthplay',10:'eleventhplay',11:'twelvthplay',
                   12:'thirteenthplay',13:'fourteenthplay',14:'fifteenthplay',15:'sixteenthplay',16:'seventeethplay',
                   17:'eigteenthplay',18:'nineteenthplay',19:'twentythplay'},inplace=True)

df = df.fillna('None')

df.to_csv("games_updated.csv")

df.info()

df["victory_status"].value_counts()

df["winner"].value_counts()
