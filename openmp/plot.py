import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("output2.txt", sep=r'\s+', engine="python")
# print(df.head())

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    df, x="newton", hue="algo", y="MIter/s", ax=ax
)

fig.savefig("output.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    df[np.logical_or(df["algo"]=="Native", df["algo"]=="Opt")], x="newton", y="MIter/s", ax=ax, hue="algo"
)
fig.savefig("output_native.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    df[np.logical_or(df["algo"]=="AVX2", df["algo"]=="Opt")], x="newton", y="MIter/s", ax=ax, hue="algo"
)
fig.savefig("output_avx.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    df[np.logical_or(df["algo"]=="Parallel", df["algo"]=="Parallel_AVX2")], x="newton", y="MIter/s", ax=ax, hue="algo"
)
fig.savefig("output_parallel.png", dpi=300, bbox_inches="tight")