import pickle
import pandas as pd
with open('macnd_grid_resuts.pkl', 'rb') as f:
        results = pickle.load(f)



scores = [i[0] for i in results ]
print(pd.Series(scores).idxmax())
print(results[1])
print(pd.Series(scores).describe() )
best = [results[x] for x in range(len(results)) if results[x][0] >= 150]
print(best)

#print(sum([7+25-42+208+60+20-23+58])/8)