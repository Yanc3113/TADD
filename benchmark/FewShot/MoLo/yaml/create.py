import os

# Create a directory to hold the folders
os.makedirs('folders', exist_ok=True)

# Generate folder names from 0001 to 0056
for i in range(1, 57):
    folder_name = f"{i:04}"  # Format as 0001, 0002, ..., 0056
    os.makedirs(os.path.join('folders', folder_name), exist_ok=True)

print("56 folders created.")
