
import os
import fitz  # PyMuPDF
import re
import csv
import ollama

# Define the function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = [page.get_text() for page in doc]
    return pages_text

# Define the function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=1000):
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        else:
            split_index = text.rfind(' ', 0, chunk_size)
            if split_index == -1:
                split_index = chunk_size
            chunks.append(text[:split_index])
            text = text[split_index:].strip()
    return chunks

# Define the function to process a chunk with the new model openchat:latest
def process_chunk_with_openchat(chunk, model='openchat:latest'):
    prompt = f"Extract if it is possible a requirement description and its respective configuration from the following text:\n{chunk}\n\nRequirement: \nConfiguration:"
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': "You are a helpful assistant that extracts requirements and device configurations from text. If it is not possible to do so answer just with NaN."},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response['message']['content']

# Define the function to extract requirements and configurations from the LLM response
def extract_requirements_and_configurations(response):
    requirement_match = re.search(r"Requirement:\s*(.*?)\n", response, re.IGNORECASE | re.DOTALL)
    configuration_match = re.search(r"Configuration:\s*(.*?)$", response, re.IGNORECASE | re.DOTALL)
    if requirement_match and configuration_match:
        requirement = requirement_match.group(1).strip()
        configuration = configuration_match.group(1).strip()
        if requirement and configuration:
            return {"requirement": requirement, "configuration": configuration}
    return None

# Define the function to save the dataset to a CSV file
def save_to_csv(data, output_file='requirements.csv'):
    file_exists = os.path.isfile(output_file)
    keys = data.keys()
    with open(output_file, 'a', newline='', encoding='utf-8') as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=keys)
        if not file_exists:
            dict_writer.writeheader()  # Write header only if file doesn't exist
        dict_writer.writerow(data)

# Function to check if the document has already been processed
def is_document_processed(doc_name, doclist_path='doclist.txt'):
    if os.path.isfile(doclist_path):
        with open(doclist_path, 'r') as doclist_file:
            processed_docs = doclist_file.read().splitlines()
            return doc_name in processed_docs
    return False

# Function to mark the document as processed
def mark_document_as_processed(doc_name, doclist_path='doclist.txt'):
    with open(doclist_path, 'a') as doclist_file:
        doclist_file.write(doc_name + '\n')

# Main function to execute the whole process
def main(directory_path, output_file='requirements.csv', model='openchat:latest', doclist_path='doclist.txt'):
    # Iterate over all PDFs in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            
            # Check if the document has already been processed
            if is_document_processed(filename, doclist_path):
                print(f"{filename} has already been processed.")
                continue
            
            print(f"Processing {filename}...")
            
            # Step 1: Extract text from PDF
            pages_text = extract_text_from_pdf(pdf_path)
            
            # Step 2: Chunk the text
            all_chunks = []
            for page_text in pages_text:
                chunks = chunk_text(page_text)
                all_chunks.extend(chunks)
            
            # Step 3 & 4: Process chunks with LLM and generate dataset
            dataset = []
            for i, chunk in enumerate(all_chunks):
                response = process_chunk_with_openchat(chunk, model)
                result = extract_requirements_and_configurations(response)
                if result:
                    print(f"Chunk {i+1}: Requirement: {result['requirement']}, Configuration: {result['configuration']}")
                    dataset.append(result)
                    save_to_csv(result, output_file)
                else:
                    print(f"Chunk {i+1}: No requirements or configurations identified.")
            
            # Mark the document as processed
            mark_document_as_processed(filename, doclist_path)
            print(f"Finished processing {filename}.")
    
    print("All documents processed.")

# Run the main function with the path to your directory containing PDFs
if __name__ == "__main__":
    directory_path = './'  # Replace with your actual directory path containing PDFs
    main(directory_path)
