# Dense Passage Retrieval – Question Answering System


## Abstract
This project presents a web-based Question Answering System that allows users to
ask questions through a browser interface and receive accurate answers from a
predefined dataset. The system is based on the concept of Dense Passage Retrieval
(DPR), where relevant information is retrieved from stored passages rather than
generated dynamically. To ensure reliable results with a small dataset, a direct
question–answer mapping approach is used. The project demonstrates key concepts
of information retrieval, backend development, and web-based interaction, making
it suitable for academic submission and viva demonstration.


## Project Overview
This project is a simple Question Answering System built using Python and Flask.
The user can open a web page, type a question, and get an answer based on the
data stored in a text file.

The project demonstrates the concept of Dense Passage Retrieval (DPR).
For accurate results with a small dataset, direct question–answer matching is used.

---

## Features
- Web-based interface to ask questions
- Instant answers from stored data
- Simple and easy dataset
- Suitable for college demo and viva

---

## Project Structure

DPR/
├── app.py
├── encode_passages.py
├── requirements.txt (optional)
├── data/
│   └── passages.txt
├── embeddings/
│   └── passage_vectors.npy
└── README.md

---

## ## Full Technology Stack

### Programming Language
- Python

### Backend Framework
- Flask

### Libraries and Tools
- NumPy – for numerical operations
- Transformers – for DPR concept implementation
- FAISS – for similarity search (conceptual usage)
- Torch (PyTorch) – for model handling
- Virtual Environment (.venv) – for dependency management

### Frontend
- HTML (basic form interface)

### Development Tools
- Visual Studio Code
- Command Line / Terminal

## Dataset
The dataset is stored in:
data/passages.txt

Each line contains a question and its answer.

Example:
what is ai : Artificial Intelligence is the ability of machines to think like humans.

---

## How to Run the Project (First Time)

1. Open the project folder in VS Code.
2. Open terminal in VS Code.

3. Create a virtual environment:
   python -m venv .venv

4. Activate the virtual environment (Windows):
   .venv\Scripts\activate

5. Install required libraries:
   pip install flask numpy torch transformers faiss-cpu

6. Run the application:
   python app.py

7. Open browser and go to:
   http://127.0.0.1:5000/

---

## How to Run After Reopening Laptop

After closing and reopening the laptop:

1. Open VS Code.
2. Open the project folder.
3. Open terminal.
4. Activate virtual environment:
   .venv\Scripts\activate
5. Run the project:
   python app.py
6. Open browser:
   http://127.0.0.1:5000/

---

## How to Transfer and Run Using ZIP File

1. Create a ZIP file of the project folder.
2. Transfer the ZIP file to another laptop.
3. Extract the ZIP file.
4. Open the extracted folder in VS Code.

5. Install Python (if not installed).
6. Create virtual environment:
   python -m venv .venv
7. Activate virtual environment:
   .venv\Scripts\activate
8. Install libraries:
   pip install flask numpy torch transformers faiss-cpu
9. Run the project:
   python app.py
10. Open browser:
    http://127.0.0.1:5000/

---

## When to Run encode_passages.py

Run encode_passages.py only when:
- passages.txt is modified
- New questions or answers are added

Command:
python encode_passages.py

---

## Viva Explanation (Short)
This project is a question answering system that retrieves answers from stored
data. It demonstrates information retrieval concepts using a simple web
interface and backend processing.

---

## Conclusion
The project successfully implements a basic and reliable Question Answering
System suitable for academic submission and demonstration.
