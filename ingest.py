import os
from src.rag import ingest_pdf

# Define where your PDFs are
DATA_FOLDER = "data"

def main():
    # 1. Check if folder exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created '{DATA_FOLDER}' folder. Please put your PDFs inside it and run this script again.")
        return

    # 2. Find all PDF files
    print(f"Scanning '{DATA_FOLDER}' for documents...")
    files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]

    if not files:
        print("No PDFs found! Please drop your exam papers in the 'data/' folder.")
        return

    # 3. Loop through and upload each one
    print(f"found {len(files)} files. Starting upload process...")
    
    for pdf_file in files:
        file_path = os.path.join(DATA_FOLDER, pdf_file)
        try:
            print(f"Processing: {pdf_file}")
            ingest_pdf(file_path)
            print(f"Finished: {pdf_file}")
        except Exception as e:
            print(f"Error uploading {pdf_file}: {e}")
            
    print("\nAll documents processed! You can now chat with your bot.")

if __name__ == "__main__":
    main()