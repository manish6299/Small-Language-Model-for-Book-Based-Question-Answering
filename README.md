# Small-Language-Model-for-Book-Based-Question-Answering
1️⃣ This project aims to build a Small Language Model (SLM) capable of extracting information from a book (PDF) and answering user queries accurately. The model leverages NLP techniques, FAISS for efficient retrieval, and a transformer-based question-answering model.

2️⃣ Approach
🔹 Data Preprocessing
Extract Text from PDF: Used pdfplumber to extract text from the uploaded book.
Text Chunking: Used RecursiveCharacterTextSplitter to split large text into manageable overlapping chunks (500 characters, 100 overlap).
Embedding Generation: Used all-MiniLM-L6-v2 from SentenceTransformers to encode text into numerical vectors.
🔹 Retrieval Mechanism
Used FAISS (Facebook AI Similarity Search) for efficient semantic search.
The query is converted into an embedding and compared against stored chunks to retrieve the most relevant sections.
🔹 Answer Generation
Transformer Model: Used distilbert-base-uncased-distilled-squad for question-answering.
Extended Response Length: Increased max_answer_len and included additional context for more detailed responses.

3️⃣ Model Architecture
The architecture consists of the following components:
PDF Text Extraction Module – Extracts raw text from the uploaded PDF.
Text Chunking & Embedding Module – Splits text into chunks and embeds them using a SentenceTransformer.
FAISS Index for Retrieval – Stores embeddings and retrieves the most relevant chunks for a given query.
Question Answering Model – Uses a transformer model to generate responses based on the retrieved text.

4️⃣ Running Instructions
🔹 Prerequisites
Ensure you have the following installed:
pip install streamlit pdfplumber faiss-cpu torch numpy sentence-transformers transformers langchain
🔹 Running the Application
Run the Streamlit app:
streamlit run app.py
🔹 Sample Input & Expected Output
Upload a PDF book (e.g., "IOT_ESE.pdf").
Ask a question: "Describe Cloud computing reference model." 
Answer: Cloud-based delivery and a network ( Internet, Intranet, Intercloud)
Additional Context: 1.Front end ( Fat client, Thin client) 2.Back-end platforms ( Servers, Storage ) 3.Cloud-based delivery and a network ( Internet, Intranet, Intercloud) ` Describe Cloud computing reference model. Infrastructure as a Service (IaaS) Key Features: ▪ Flexibility and Control: IaaS provides virtualized resources like Virtual Machines (VMs), storage, and networks, giving users full control over operating...

5️⃣ Key Learnings & Observations
✅ Efficient Retrieval: FAISS indexing significantly improves the speed of retrieving relevant text.
✅ Improved Answer Quality: Increasing retrieved chunks and max_answer_len enhances answer quality.
✅ Scalability: The model can be deployed on Hugging Face Spaces or converted into an API using FastAPI.
✅ Future Enhancements: Fine-tuning a larger model (e.g., Mistral-7B) could improve answer accuracy.
