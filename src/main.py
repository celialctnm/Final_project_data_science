import pandas as pd

# process data
music_data = pd.read_csv("../data/train.csv")
print(music_data.head(5))

music_features = music_data.drop(columns=['ID', 'emotion'])
music_target = music_data['emotion']