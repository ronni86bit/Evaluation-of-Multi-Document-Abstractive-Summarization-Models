import pandas as pd
import matplotlib.pyplot as plt
results = [
    {
        "Model": "BART",
        "ROUGE-1": 44.2,
        "ROUGE-2": 21.3,
        "ROUGE-L": 41.1,
        "Training Time (hrs)": 5.2,
        "GPU/TPU": "NVIDIA A100"
    },
    {
        "Model": "PEGASUS",
        "ROUGE-1": 45.8,
        "ROUGE-2": 22.5,
        "ROUGE-L": 42.6,
        "Training Time (hrs)": 6.5,
        "GPU/TPU": "NVIDIA A100"
    },
    {
        "Model": "T5",
        "ROUGE-1": 42.7,
        "ROUGE-2": 20.1,
        "ROUGE-L": 39.9,
        "Training Time (hrs)": 4.8,
        "GPU/TPU": "NVIDIA V100"
    }
]
df = pd.DataFrame(results)

print("\n=== Model Comparison Table ===\n")
print(df.to_string(index=False))
metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df[metric], color="skyblue")
    plt.title(f"{metric} Score Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.ylim(0, max(df[metric]) + 5)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if "Training Time (hrs)" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["Training Time (hrs)"], color="lightgreen")
    plt.title("Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Training Time (hrs)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
