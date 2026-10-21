
import asyncio
import re
from typing import Any, Dict, List

import streamlit as st

from rag.core import run_llm
from rag.loader import upload_file
from ingestion import index_document_async


st.set_page_config(
    page_title="AskMyDocs Helper",
    layout="centered",
)

st.title("Welcome to AskMyDocs")


# Format Markdown and LaTeX in LLM responses
def format_response(text: str) -> str:
    """Convert common LaTeX delimiters into Streamlit-friendly Markdown."""

    # Block equations: \[ ... \] -> $$ ... $$
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "$$\n" + match.group(1).strip() + "\n$$",
        text,
        flags=re.DOTALL,
    )

    # Inline equations: \( ... \) -> $ ... $
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: "$" + match.group(1).strip() + "$",
        text,
        flags=re.DOTALL,
    )

    return text


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "AskMyDocs can help you!",
        }
    ]


def format_source(context_docs: List[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(
                (getattr(doc, "metadata", None) or {}).get("source")
                or "Unknown"
            )
            for doc in (context_docs or [])
        )
    )


# Sidebar
with st.sidebar:
    st.subheader("Session")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "AskMyDocs can help you!",
            }
        ]
        st.rerun()

    st.divider()

    # Upload document
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx"],
    )

    if st.button("Process Document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
        else:
            try:
                with st.spinner("Loading and indexing document..."):
                    # 1. Load document
                    docs = upload_file(uploaded_file)

                    if not docs:
                        st.warning(
                            "No content was extracted from the file."
                        )
                    else:
                        # 2. Split, create embeddings, and index
                        count = asyncio.run(
                            index_document_async(docs)
                        )

                        st.success(
                            f"Successfully indexed {count} chunks "
                            f"from {uploaded_file.name}."
                        )

                        with st.expander("Preview document"):
                            st.write(docs[0].page_content[:1000])

                        with st.expander("Indexing details"):
                            st.write(f"Documents loaded: {len(docs)}")
                            st.write(f"Chunks indexed: {count}")

            except Exception as e:
                st.error(f"Failed to process document: {e}")


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(format_response(msg["content"]))

        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")


# Chat input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Retrieving documents and generating answer..."
            ):
                result: Dict[str, Any] = run_llm(prompt)

                answer = (
                    str(result.get("answer", "")).strip()
                    or "(No answer returned)"
                )

                sources = format_source(
                    result.get("context", [])
                )

            # Format the response before displaying
            st.markdown(format_response(answer))

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as e:
            error_message = f"Failed to generate response: {e}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
