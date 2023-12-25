import os
import pandas as pd

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

data = {
    "Message": ["This is the first test caption", "This is the second test caption"]
}

df = pd.DataFrame(data)
df.to_csv("./Data/captions.csv", index=True)
