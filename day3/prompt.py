from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class PromptGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_retrival_grader(self):
        system = (
            "You are a grader assessing relevance "
            "of a retrieved document to a user question. If the document contains keywords related to the user question, "
            "grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n"
            "Provide the binary score as a JSON with a single key 'score' and no premable or explanation."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "question: {question}\n\n document: {document} "),
            ]
        )
        return prompt | self.llm | JsonOutputParser()
    
    def generate_rag_chain(self):
        system = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "question: {question}\n\n context: {context} "),
            ]
        )
        return prompt | self.llm | StrOutputParser()
    
    def generate_hallucination_grader(self):
        system = (
            "You are a grader assessing whether "
            "an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate "
            "whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a "
            "single key 'score' and no preamble or explanation."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "documents: {documents}\n\n answer: {generation} "),
            ]
        )
        return prompt | self.llm | JsonOutputParser()
    