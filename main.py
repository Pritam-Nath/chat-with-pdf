import torch
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PDF file
loader = PyPDFLoader(file_path=r".\Sachin.pdf")
data = loader.load()

# Split large text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

# Load sentence-transformer model for embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Create Chroma vectorstore
vector_store = Chroma.from_documents(text_chunks, embedding=embeddings)

# Load the LLaMA.cpp model (make sure this .gguf file is in the same folder)
llm_answer_gen = LlamaCpp(
    model_path=r"./mistral-7b-openorca.Q4_0.gguf",
    temperature=0.75,
    top_p=1,
    f16_kv=True,
    n_ctx=4096,
    n_gpu_layers=40,     # adjust based on your GPU memory
    n_batch=512,         # tuning batch size
    verbose=False
)

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the RAG pipeline
answer_gen_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_answer_gen,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Start interactive loop
print("\n🤖 Ask anything from the PDF (type 'q' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower().strip() == 'q':
        print("Goodbye!")
        break

    answer = answer_gen_chain.run({"question": user_input})
    print("Bot:", answer)
