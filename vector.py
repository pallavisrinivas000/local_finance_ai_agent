from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

df = pd.read_csv("aug_expenses.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
    # Skip rows with missing date or narration
        if pd.isna(row["Date"]) or pd.isna(row["Narration"]):
            continue

        # Determine transaction type and amount
        if pd.notna(row["Withdrawal Amt."]):
            txn_type = "Withdrawn"
            amount = row["Withdrawal Amt."]
        elif pd.notna(row["Deposit Amt."]):
            txn_type = "Deposited"
            amount = row["Deposit Amt."]
        else:
            txn_type = "No transaction"
            amount = 0

        page_content = (
            f"On {row['Date']}, {row['Narration']}: {txn_type} ₹{amount}. "
            f"Closing balance: ₹{row['Closing Balance']}."
        )
        metadata = {
            "date": row["Date"],
            "narration": row["Narration"],
            "ref_no": row["Chq./Ref.No."],
            "value_date": row["Value Dt"],
            "withdrawal_amt": row["Withdrawal Amt."],
            "deposit_amt": row["Deposit Amt."],
            "closing_balance": row["Closing Balance"],
        }
        document = Document(page_content=page_content, metadata=metadata, id=str(i))
        ids.append(str(i))
        documents.append(document)

    
vector_store = Chroma(
    collection_name="aug_expenses",
    embedding_function=embeddings,
    persist_directory=db_location,
)

if add_documents:
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
    } 
)


# Print the number of documents in the vector database
print("Total documents in vector DB:", len(vector_store.get()['ids']))

# Print the first 3 documents and their metadata
docs = vector_store.get()
for i in range(min(100, len(docs['ids']))):
    print(f"\nDocument {i+1}:")
    print("ID:", docs['ids'][i])
    print("Page Content:", docs['documents'][i])