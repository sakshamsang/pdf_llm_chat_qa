PDF LLM Chat Q&A App
<img alt="Streamlit Banner" src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png">

🚀 Overview
PDF LLM Chat Q&A is a modern Streamlit application that lets you:

📄 Upload any PDF document
🔍 Instantly extract and index text and tables
🤖 Ask questions about your PDF using a powerful LLM (Mistral)
⚡ Get fast, context-aware answers without scanning the PDF every time
🛠️ Features
PDF Upload: Drag & drop your PDF file
Automatic Extraction: Text and tables are extracted and cached
Excel Cache: Each PDF is saved as an Excel file for fast future access
LLM Q&A: Ask anything about your PDF, answers are generated from cached Excel data
Industry Standard: Designed for scalability and real-world use
💻 Usage Instructions
1. Clone the Repository
2. Install Dependencies
Activate your Python virtual environment and install required packages:

3. Run the Streamlit App
4. Use the App
Step 1: Upload your PDF
Step 2: Wait for extraction and caching (Excel file is created)
Step 3: Type your question in the chat box
Step 4: Get instant, smart answers powered by Mistral LLM
📊 How It Works
PDF Extraction: Text and tables are extracted using pdfplumber and saved to Excel
Excel as Database: Each PDF's data is cached in an Excel file (one sheet per table/section)
LLM Q&A: When you ask a question, the app loads data from Excel, chunks it, and uses Mistral LLM to answer
🖼️ Screenshots
<img alt="PDF Upload" src="https://user-images.githubusercontent.com/streamlit-upload-demo.png">

<img alt="Q&amp;A Chat" src="https://user-images.githubusercontent.com/streamlit-chat-demo.png">

🏗️ Roadmap
 PDF upload & extraction
 Excel caching
 LLM Q&A
 Advanced table extraction
 User authentication
 Cloud deployment
🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📧 Contact
For questions or support, reach out to Saksham Sang