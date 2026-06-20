import pandas as pd

# Only the 4 real columns are needed; the rest are unused metadata in the raw CSV
df = pd.read_csv("data/Resume.csv", usecols=['ID', 'Resume_str', 'Resume_html', 'Category'])

print("=" * 50)
print("BEFORE CLEANING")
print("=" * 50)
print(f"Total rows        : {len(df)}")
print(f"Missing Resume_str: {df['Resume_str'].isnull().sum()}")
print(f"Missing Category  : {df['Category'].isnull().sum()}")

# Drop rows where Resume_str or Category is missing
df = df.dropna(subset=['Resume_str', 'Category'])

# Clean whitespace from Category column
df['Category'] = df['Category'].str.strip()

# Real categories are short labels (e.g. "HR", "ENGINEERING"); filter out
# any row where Category accidentally contains resume text instead of a label
df = df[df['Category'].str.len() < 40]

# Clean Resume_str and drop resumes that are too short to be useful
df['Resume_str'] = df['Resume_str'].str.strip()
df = df[df['Resume_str'].str.len() > 100]

df = df.reset_index(drop=True)

print("\n" + "=" * 50)
print("AFTER CLEANING")
print("=" * 50)
print(f"Total clean rows  : {len(df)}")
print(f"Unique categories : {df['Category'].nunique()}")

print("\n" + "=" * 50)
print("CATEGORIES AND COUNTS")
print("=" * 50)
print(df['Category'].value_counts())

print("\n" + "=" * 50)
print("SAMPLE CLEAN RESUME (first 300 chars)")
print("=" * 50)
print(df['Resume_str'].iloc[0][:300])

df[['Resume_str', 'Category']].to_csv("data/Resume_clean.csv", index=False)
print("\nClean dataset saved to data/Resume_clean.csv")
print(f"Ready to train on {len(df)} resumes across {df['Category'].nunique()} categories")