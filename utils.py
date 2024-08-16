from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import  OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def qa_agent(openai_api_key, memory ,uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       api_key=openai_api_key,
                       base_url="https://api.chatanywhere.tech/v1")

    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1111,  # 每块文本的最大长度
        chunk_overlap=50,  # 分割片段之间的重叠长度
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                        api_key=openai_api_key,
                                        base_url="https://api.chatanywhere.tech/v1")
    db = FAISS.from_documents(texts,embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = qa.invoke({"chat_history":memory,"question":question})
    return  response
