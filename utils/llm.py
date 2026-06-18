import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Support both .env (local) and Streamlit Cloud secrets
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("❌ GOOGLE_API_KEY not found. Set it in .env or Streamlit Secrets.")
        st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


def rewrite_question(question, history):

    conversation = ""

    for msg in history[-6:]:

        conversation += (
            f"{msg['role']} : "
            f"{msg['content']}\n"
        )

    prompt = f"""

Rewrite the user's latest question into a standalone question.

Conversation:

{conversation}

Current Question:

{question}

Return ONLY the rewritten question.

"""

    response = model.generate_content(prompt)

    return response.text.strip()


def build_prompt(
        context,
        question,
        history
):

    conversation = ""

    for msg in history[-6:]:

        conversation += (
            f"{msg['role']} : "
            f"{msg['content']}\n"
        )

    prompt = f"""

You are a helpful PDF assistant.

Previous Conversation:

{conversation}


Retrieved Context:

{context}


Current Question:

{question}


Rules:

1. Use previous conversation to understand:

- this
- that
- it
- explain again
- simplify it

2. Answer ONLY from context.

3. If answer not found:

"I could not find the answer in the uploaded PDFs."

Answer:

"""

    return prompt


def generate_answer(
        results,
        question,
        history
):

    contexts = results["documents"][0]

    context_text = "\n\n".join(contexts)

    prompt = build_prompt(
        context_text,
        question,
        history
    )

    response = model.generate_content(prompt)

    citations = []

    for i, meta in enumerate(results["metadatas"][0]):

        citations.append(

            {

                "source": meta["source"],

                "page": meta["page"],

                "excerpt": contexts[i][:250]

            }

        )

    return response.text, citations