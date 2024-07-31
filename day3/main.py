import streamlit as st
from agent import RagAgent
from dotenv import load_dotenv
from prompt import PromptGenerator
from retriever import Retriever

load_dotenv(dotenv_path="../.env", verbose=True)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
retriever = Retriever(urls).create()

prompt_generator = PromptGenerator()
retrieval_grader = prompt_generator.generate_retrival_grader()
rag_chain = prompt_generator.generate_rag_chain()
hallucination_grader = prompt_generator.generate_hallucination_grader()

app = RagAgent(retriever, retrieval_grader, rag_chain, hallucination_grader).compile()

# Streamlit 앱 UI
st.title("Research Assistant powered by OpenAI")

input_topic = st.text_input(
    ":female-scientist: Enter a topic",
    value="Superfast Llama 3 inference on Groq Cloud",
)

generate_report = st.button("Generate Report")

def to_final_report(value):
    if "generation" not in value:
        return "## Answer\nFailed to generation"

    final_report = "## Answer\n"
    final_report += value["generation"]
    final_report += "\n"

    final_report += "## Metadata\n"
    if value["web_search"]:
        final_report += "### From web search\n"
    else:
        final_report += "### From retrieve\n"

    documents = value["documents"]
    for doc in documents:
        final_report += f"- {doc.metadata['title']} ({doc.metadata['source']})\n"
    
    return final_report

if generate_report:
    with st.spinner("Generating Report"):
        inputs = {"question": input_topic}
        for output in app.stream(inputs):
            for key, value in output.items():
                pass
        st.markdown(to_final_report(value))

st.sidebar.markdown("---")
if st.sidebar.button("Restart"):
    st.session_state.clear()
    st.experimental_rerun()
