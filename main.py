from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="mistral:7b-instruct-q4_0")

template = """
You are an expert in answering question about a person's monthly expenses summary

Here are some records of monthl y expenses:{records}

Here is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


while True:
    question = input("Enter your question about monthly expenses (or type 'exit' to quit): ")

    if question.lower() == 'exit':
        break

    records = retriever.invoke(question) 

    result = chain.invoke({"records": records, "question": question})
    
    print(result)