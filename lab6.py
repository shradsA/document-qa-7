# lab6.py
import streamlit as st
from openai import OpenAI
import json
import re

# ------------------------------
# HELPER FUNCTION: Robust JSON parsing
# ------------------------------
def parse_model_output(output_text):
    """
    Attempt to convert model output into JSON.
    Handles cases where model returns Markdown or citations.
    """
    # First, try to extract JSON directly
    json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    # Fallback: return structured object with plain text explanation
    return {
        "Claim": "Unknown",
        "Verdict": "Unverified",
        "Confidence": "50",
        "Explanation": output_text.strip(),
        "Sources": []
    }


# ------------------------------
# FACT-CHECK FUNCTION
# ------------------------------
def fact_check_claim(claim):
    """Fact-check a given claim using the OpenAI Responses API with web_search."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    system_prompt = """
    You are an AI Fact-Checker and Citation Builder.
    Your job is to verify factual claims using current, credible information.
    Use the web_search tool to find evidence.
    ALWAYS return a valid JSON object in this exact format:
    {
        "Claim": "string",
        "Verdict": "True / False / Partially True / Unverified",
        "Confidence": "0-100",
        "Explanation": "string",
        "Sources": [{"title": "string", "url": "string"}]
    }
    Confidence score represents your certainty level (based on source agreement).
    DO NOT include any text outside the JSON object.  
    If unsure, leave fields empty rather than adding extra commentary.
    """

    # Call OpenAI Responses API (without unsupported reasoning)
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Fact-check the following claim: {claim}"}
        ]
    )

    # Extract text output
    output_text = ""
    for item in response.output:
        if item.type == "message":
            for content_item in item.content:
                if content_item.type == "output_text":
                    output_text += content_item.text

    # Parse JSON or fallback
    result = parse_model_output(output_text)
    return result


# ------------------------------
# STREAMLIT APP
# ------------------------------
def run():
    st.title("🧠 AI Fact-Checker + Citation Builder")

    # Initialize session history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input for factual claim
    user_claim = st.text_input("Enter a factual claim:")

    if st.button("Check Fact"):
        if not user_claim.strip():
            st.warning("Please enter a claim before checking.")
        else:
            with st.spinner("Verifying..."):
                result = fact_check_claim(user_claim)

                # Display structured JSON
                st.subheader("✅ Fact-Check Result")
                st.json(result)

                # Display formatted summary
                st.markdown(f"**🗣️ Claim:** {result.get('Claim','Unknown')}")
                st.markdown(f"**📊 Verdict:** {result.get('Verdict','Unverified')}")
                st.markdown(f"**🎯 Confidence:** {result.get('Confidence','50')}%")
                st.markdown(f"**🧾 Explanation:** {result.get('Explanation','No explanation provided')}")

                # Clickable sources
                if "Sources" in result and isinstance(result["Sources"], list) and result["Sources"]:
                    st.markdown("### 🔗 Sources")
                    for src in result["Sources"]:
                        st.markdown(f"- [{src.get('title','Unknown')}]({src.get('url','#')})")

                # Save to session history
                st.session_state.history.append(result)

    # Display session history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("🕓 Session History")
        for i, record in enumerate(reversed(st.session_state.history), 1):
            st.markdown(
                f"**{i}. {record.get('Claim', 'Unknown Claim')}** → "
                f"{record.get('Verdict', 'N/A')} ({record.get('Confidence', 'N/A')}%)"
            )

    # Reflection and Discussion
    st.markdown("---")
    st.subheader("💬 Reflection & Discussion")
    st.markdown("""
    **1. How did the model’s reasoning feel different from a standard chat model?**  
    → The Responses API uses structured reasoning and live web search, producing factual and evidence-based responses rather than just conversational replies.

    **2. Were the sources credible and diverse?**  
    → The AI attempts to cite reputable sources (.edu, .gov, or trusted news sites) to increase reliability.

    **3. How does tool integration enhance trust and accuracy?**  
    → Integrating the web_search tool ensures answers are grounded in real-time information, improving trust and verification.
    """)


if __name__ == "__main__":
    run()

