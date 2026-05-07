from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


INPUT_FILE = "input.txt"
DB_DIR = "./chroma_db"

#LLM_MODEL = "qwen2.5:7b"
#LLM_MODEL = "gemma4:26b"
#LLM_MODEL = "gpt-oss:20b"
LLM_MODEL = "qwen3:4b"
EMBED_MODEL = "nomic-embed-text"



def load_input_txt():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    return [Document(page_content=text)]


def build_vector_db():
    docs = load_input_txt()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return vector_db


def main():
    print("input.txt 인덱싱 중...")
    vector_db = build_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.2,
        num_ctx=8192
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
너는 로컬 문서 기반 질의응답 AI다.
반드시 아래 문서 내용(Context)을 우선 근거로 답변하라.
문서에 없는 내용은 '문서에서 확인되지 않습니다'라고 말하라.

Context:
{context}
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chat_history = []

    print("\n로컬 RAG 챗봇 시작. 종료하려면 exit 입력\n")

    while True:
        question = input("User: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = prompt.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })

        answer = llm.invoke(messages).content

        print("\nAI:", answer, "\n")

        # 최소 대화기억
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # 너무 길어지는 것 방지
        chat_history = chat_history[-10:]


if __name__ == "__main__":
    main()
