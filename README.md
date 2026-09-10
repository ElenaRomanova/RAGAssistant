This project is a RAG (Retrieval-Augmented Generation) system that provides information about the normative legal acts of Republic of Kazakhstan. The system utilizes a combination of retrieval techniques and generative models to deliver accurate and relevant legal information to users.
There are two chunking strategies implemented in this project: structured chunking and overlapping chunking.

To ask a question, you can use the following command:

python -m rag.rag_traced.py "Your question here"


To run the chat UI you can use the following command:

python -m rag.chatbot.py


To run the evaluation on questions test set you can use the following command:

python -m eval.run.py
Default strategy for evaluation is structured chunking. 