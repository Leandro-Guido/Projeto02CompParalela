import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

x = "Frequency"
y = "Monetary"

# Before clustering
# df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
# df = df[[x, y]]
# df = df.dropna()

# sns.scatterplot(x=df[x], y=df[y])
# plt.title(f"Scatterplot of {y} (y) vs {x} (x)")

# After clustering
plt.figure()
df = pd.read_csv("output.csv")
sns.scatterplot(x=df.Frequency, y=df.Monetary, 
                hue=df.Cluster, 
                palette=sns.color_palette("hls", n_colors=5))
plt.xlabel(x)
plt.ylabel(y)
plt.title(f"Clustered: {y} (y) vs {x} (x)")

plt.show()