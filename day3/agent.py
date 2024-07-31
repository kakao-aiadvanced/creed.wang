import os
from typing import List

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from typing_extensions import TypedDict


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    retry_count: int
    documents: List[str]


class RagAgent:
    def __init__(self, retriever, retrieval_grader, rag_chain, hallucination_grader):
        self.retriever = retriever
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.retrieval_grader = retrieval_grader
        self.rag_chain = rag_chain
        self.hallucination_grader = hallucination_grader
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("docs_retrieval", self._docs_retrieval)
        workflow.add_node("relevance_checker", self._relevance_checker)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("search_tavily", self._search_tavily)

        workflow.set_entry_point("docs_retrieval")
        workflow.add_edge("docs_retrieval", "relevance_checker")
        workflow.add_conditional_edges(
            "relevance_checker",
            self._decide_to_generate,
            {"websearch": "search_tavily", "generate": "generate_answer", "failed": END}
        )
        workflow.add_edge("search_tavily", "relevance_checker")
        workflow.add_conditional_edges(
            "generate_answer",
            self._hallucination_checker,
            {"success": END, "failed": "generate_answer"}
        )
        return workflow

    def _docs_retrieval(self, state):
        print("1. DOCS RETRIEVE")
        question = state["question"]

        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question, "web_search": False}

    def _relevance_checker(self, state):
        print("2. RELEVANCE CHECKER")
        documents = state["documents"]
        question = state["question"]

        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score["score"]
            if grade.lower() == "yes":
                filtered_docs.append(d)
        return {"documents": filtered_docs, "question": question}

    def _decide_to_generate(self, state):
        print("3. DECIDE TO GENERATE")
        documents = state["documents"]
        retry_count = state.get("retry_count")
        retry_count = 0 if retry_count is None else retry_count

        if retry_count > 2:
            print("  FAILED TO GENERATE")
            return "failed"

        if len(documents) == 0:
            print("  USE WEB SEARCH")
            return "websearch"
        else:
            print("  GO TO GENERATE")
            return "generate"

    def _search_tavily(self, state):
        print("3-1. SEARCH TAVILY")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        retry_count = 0 if retry_count is None else retry_count

        search_results = self.tavily.search(query=question, max_results=3)['results']
        documents = []
        for search_result in search_results:
            document = Document(
                page_content=search_result["content"],
                metadata={"source": search_result["url"], "title": search_result["title"]}
            )
            documents.append(document)
        retry_count += 1
        return {"documents": documents, "question": question, "web_search": True, "retry_count": retry_count}

    def _generate_answer(self, state):
        print("4. GENERATE ANSWER")
        question = state["question"]
        documents = state["documents"]
        web_search = state["web_search"]

        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation, "web_search": web_search}

    def _hallucination_checker(self, state):
        print("5. HALLUCINATION CHECKER")
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        if grade == "yes":
            print("  NO HALLUCINATION")
            return "success"
        else:
            print("  HALLUCINATION FOUND")
            return "failed"

    def compile(self):
        return self.workflow.compile()
