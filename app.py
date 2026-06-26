import streamlit as st
from agent.ideas import generate_ideas
from agent.retriever import retrieve, retrieve_with_scores
from agent.generator import stream_draft
from agent.improver import improve
from agent.researcher import research
from agent.finalizer import stream_article, generate_image_prompt

st.set_page_config(page_title="AI Content Assistant", page_icon="✍️", layout="wide")
st.title("AI Content Assistant")
st.caption("Generate high-quality articles powered by a knowledge base + web research.")

# ── Session state init ────────────────────────────────────────────────────────
for key in ("ideas", "selected_idea", "pipeline_done", "result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Step 1: Topic input ───────────────────────────────────────────────────────
st.header("1. Enter Your Topic")
topic = st.text_input("What do you want to write about?", placeholder="e.g. software testing best practices")

if st.button("Generate Ideas", disabled=not topic):
    st.session_state.ideas = None
    st.session_state.selected_idea = None
    st.session_state.pipeline_done = None
    st.session_state.result = None
    with st.spinner("Generating content ideas..."):
        st.session_state.ideas = generate_ideas(topic)

# ── Step 2: Idea selection ────────────────────────────────────────────────────
if st.session_state.ideas:
    st.header("2. Select an Idea")
    selected = st.radio(
        "Choose the article idea you want to develop:",
        st.session_state.ideas,
        index=0,
    )

    if st.button("Start Writing"):
        st.session_state.selected_idea = selected
        st.session_state.pipeline_done = False
        st.session_state.result = None

# ── Step 3–7: Pipeline execution (streamed) ───────────────────────────────────
if st.session_state.selected_idea and st.session_state.pipeline_done is False:
    idea = st.session_state.selected_idea
    st.header("3. Generating Your Article")
    st.info(f"**Selected idea:** {idea}")

    with st.spinner("🔍 Retrieving knowledge base context..."):
        kb_chunks_scored = retrieve_with_scores(idea)
        kb_chunks = [text for text, _ in kb_chunks_scored]

    st.subheader("✍️ Drafting article")
    draft = st.write_stream(stream_draft(idea, kb_chunks))

    with st.status("Refining...", expanded=True) as status:
        st.write("🔄 Evaluating and improving...")
        improved_draft, evals = improve(draft, kb_chunks)

        st.write("🌐 Researching web sources...")
        research_result = research(idea)
        status.update(label="Refinement complete!", state="complete")

    st.header("4. Your Article")
    article = st.write_stream(stream_article(idea, improved_draft, research_result["summary"]))

    with st.spinner("🎨 Generating image prompt..."):
        image_prompt = generate_image_prompt(idea)

    st.session_state.result = {
        "kb_chunks": kb_chunks_scored,
        "draft": draft,
        "evals": evals,
        "research": research_result,
        "article": article,
        "image_prompt": image_prompt,
    }
    st.session_state.pipeline_done = True
    st.rerun()

# ── Final output (persistent render after streaming run) ──────────────────────
if st.session_state.pipeline_done and st.session_state.result:
    result = st.session_state.result
    st.header("3. Selected Idea")
    st.info(f"**{st.session_state.selected_idea}**")

    with st.expander(f"KB Context ({len(result['kb_chunks'])} chunks retrieved)"):
        for i, (chunk, score) in enumerate(result["kb_chunks"], 1):
            st.markdown(f"**Chunk {i}** — relevance score: `{score:.4f}`")
            st.markdown(chunk)
            if i < len(result["kb_chunks"]):
                st.divider()

    with st.expander("Initial Draft"):
        st.markdown(result["draft"])

    with st.expander(f"Evaluation Results ({len(result['evals'])} round(s))"):
        for i, ev in enumerate(result["evals"], 1):
            st.markdown(f"**Round {i}** — Average score: `{ev['average']}/5`")
            cols = st.columns(3)
            for j, (criterion, score) in enumerate(ev["scores"].items()):
                cols[j % 3].metric(criterion, f"{score}/5")
            st.markdown(f"**Reasoning:** {ev['reasoning']}")

    with st.expander("Web Research"):
        st.markdown(f"**Queries:** {', '.join(result['research']['queries'])}")
        st.markdown(f"**Summary:**\n\n{result['research']['summary']}")

    st.header("4. Your Article")
    st.markdown(result["article"])

    st.divider()
    st.subheader("Image Prompt")
    st.code(result["image_prompt"], language=None)

    st.download_button(
        label="Download Article",
        data=result["article"],
        file_name="article.md",
        mime="text/markdown",
    )
