import pandas as pd

# Read the CSV file
file_path = './dataset/cic_iot_2023_demo.csv'
df = pd.read_csv(file_path)

# Get the last column (which contains the labels)
labels = df.iloc[:, -1]

# Count the occurrences of each label
label_counts = labels.value_counts()

# Display the results
print("Label counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Optional: Show total count
print(f"\nTotal samples: {len(df)}")