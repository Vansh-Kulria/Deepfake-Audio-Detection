import kagglehub

print("Downloading dataset 'mohammedabdeldayem/the-fake-or-real-dataset' via kagglehub...")
try:
    path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")
    print("SUCCESS: Path to dataset files:", path)
except Exception as e:
    print("ERROR downloading dataset:", e)
