"""
Ultra-Simple Streamlit UI - Research & Both
PDF download, inline PDF viewer, audio playback and download.
"""

import base64
import os
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

# ============================================================================
# CONFIG
# ============================================================================

# In Docker: set ORCHESTRATOR_URL=http://orchestrator:8000 via docker-compose env
# Locally:   defaults to localhost:8000
API_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
POLL_INTERVAL = 2
MAX_POLL = 450  # 450 * 2 seconds = 900 seconds = 15 minutes (research can take 5-10 min)

st.set_page_config(
    page_title="Research & Audio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = None  # "research" or "both"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

BASE_DIR = Path(__file__).parent


def check_api_health() -> bool:
    """Check if FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_task(task_id: str) -> dict:
    try:
        response = requests.get(f"{API_URL}/tasks/{task_id}", timeout=10)
        return response.json()
    except requests.exceptions.Timeout:
        # Timeout on poll is OK — server is busy, just retry
        return {"status": "in_progress"}
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API on localhost:8000 - is it running?")
        return None
    except Exception as e:
        return {"status": "in_progress"}  # Assume still running


def poll_task(task_id: str):
    progress_bar = st.progress(0)
    status_text = st.empty()
    connect_errors = 0

    for attempt in range(MAX_POLL):
        task = check_task(task_id)

        if task is None:
            connect_errors += 1
            if connect_errors >= 3:
                status_text.error("❌ Cannot connect to API — is localhost:8000 running?")
                progress_bar.empty()
                return None
            # Retry silently
            time.sleep(POLL_INTERVAL)
            continue

        connect_errors = 0  # Reset on success
        status = task.get("status", "pending")
        elapsed = attempt * POLL_INTERVAL
        progress = min(int((elapsed / 300) * 100), 99)  # Estimate over 5 min
        progress_bar.progress(progress)

        # Show meaningful status messages
        if elapsed < 30:
            msg = f"⏳ Starting research... ({elapsed}s)"
        elif elapsed < 90:
            msg = f"🔍 Searching web & generating report... ({elapsed}s)"
        elif elapsed < 150:
            msg = f"🎙️ Creating audio podcast... ({elapsed}s)"
        else:
            msg = f"⏳ Still running, please wait... ({elapsed}s)"

        status_text.info(msg)

        if status == "completed":
            progress_bar.progress(100)
            status_text.success("✅ Done!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            return task

        if status == "failed":
            error = task.get("error", "Unknown error")
            status_text.error(f"❌ Failed: {error}")
            progress_bar.empty()
            return task

        time.sleep(POLL_INTERVAL)

    progress_bar.empty()
    status_text.error("⏱️ Timeout — task is still running in background")
    return None


def resolve_file(path_str: str) -> tuple[bytes | None, Path | None]:
    """Try several path resolutions; return (bytes, resolved_path) or (None, None)."""
    candidates = [
        Path(path_str),
        BASE_DIR / path_str,
        BASE_DIR / path_str.replace("/", "\\"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_bytes(), p
        except Exception:
            pass
    return None, None


def show_pdf_inline(pdf_bytes: bytes):
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="650px" style="border:none;border-radius:6px;"></iframe>',
        unsafe_allow_html=True,
    )


def display_research_result(result: dict):
    col1, col2 = st.columns(2)
    col1.metric("📌 Topic", result.get("topic", "N/A"))
    col2.metric("⭐ Quality Score", f"{result.get('score', 0):.1f} / 100")

    pdf_path_str = result.get("file", "")
    if pdf_path_str:
        st.markdown("---")
        pdf_bytes, resolved = resolve_file(pdf_path_str)
        filename = Path(pdf_path_str).name

        if pdf_bytes:
            col_dl, col_view = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
            with col_view:
                st.caption(f"File: `{resolved}`")

            with st.expander("👁️ View PDF Report inline", expanded=True):
                show_pdf_inline(pdf_bytes)
        else:
            st.warning(f"⚠️ PDF not found at `{pdf_path_str}` — it may still be generating.")
    else:
        st.info("No PDF path returned from API.")


def display_podcast_result(result: dict):
    audio_path_str = result.get("audio", "")
    questions = result.get("questions", [])
    answers = result.get("answers", [])

    # Display single podcast audio player
    if audio_path_str:
        st.markdown("*Single audio file with all Q&A - 🇮🇳 Natural Indian English voice*")

        audio_bytes, resolved = resolve_file(audio_path_str)
        if audio_bytes:
            # Determine format
            audio_format = "audio/wav" if audio_path_str.endswith(".wav") else "audio/mp3"

            # Auto-detect format from file extension
            if audio_path_str.endswith(".mp3"):
                audio_format = "audio/mp3"
                download_label = "📥 Download Podcast (MP3)"
            elif audio_path_str.endswith(".wav"):
                audio_format = "audio/wav"
                download_label = "📥 Download Podcast (WAV)"
            else:
                audio_format = "audio/mp3"  # Default to MP3
                download_label = "📥 Download Podcast"

            # Audio player
            st.audio(audio_bytes, format=audio_format)

            # Download button
            st.download_button(
                label=download_label,
                data=audio_bytes,
                file_name=resolved.name,
                mime=audio_format,
            )

            st.success(f"✅ Podcast ready! ({resolved.name})")
        else:
            st.warning(f"⚠️ Audio file not found: {audio_path_str}")

    # Display Q&A pairs below audio
    if questions and answers:
        st.divider()
        st.markdown("### 📋 Q&A Reference")
        st.markdown(f"*{len(questions)} question & answer pairs from the podcast*")

        for i, (q, a) in enumerate(zip(questions, answers), 1):
            with st.expander(f"**Q{i}: {q[:60]}...**"):
                st.markdown(f"**Question {i}:**\n\n{q}")
                st.markdown(f"**Answer {i}:**\n\n{a}")

    elif not audio_path_str:
        st.info("No podcast or Q&A data available.")


# ============================================================================
# TITLE
# ============================================================================

st.title("🎙️ Research & Audio Generator")
st.markdown("**Research any topic — get a PDF report and an audio podcast**")
st.divider()

# ============================================================================
# INPUT + BUTTONS
# ============================================================================

topic = st.text_input(
    "📝 Enter a topic to research:",
    placeholder="e.g., Artificial Intelligence, Nemoclaw, Healthcare Technology...",
    key="topic_input",
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    research_btn = st.button("📄 Research Only", use_container_width=True, key="research_btn")
with col2:
    both_btn = st.button("🎙️ Both", use_container_width=True, key="both_btn",
                         help="PDF report AND audio podcast with Q&A")
with col3:
    clear_btn = st.button("🧹 Clear All", use_container_width=True, key="clear_btn")

# Check API health
if research_btn or both_btn:
    if not check_api_health():
        st.error("""
❌ **FastAPI server is not running!**

Start all 4 services in separate terminals:

**Terminal 1:** `python fastapi_clean.py`
**Terminal 2:** `cd researcher && python fastapi_researcher.py`
**Terminal 3:** `cd Podcast_agent && python fastapi_podcast.py`
**Terminal 4:** (This window - Streamlit)

See `00_START_ALL_SERVICES.md` for detailed instructions.
        """)
        st.stop()

if clear_btn:
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.last_mode = None
    st.rerun()

# ============================================================================
# RESEARCH ONLY
# ============================================================================

if research_btn:
    if not topic.strip():
        st.error("❌ Please enter a topic")
    else:
        st.divider()
        st.subheader("📄 Research Report")
        st.session_state.messages.append({
            "role": "user",
            "content": f"Research on: {topic}",
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        try:
            resp = requests.post(
                f"{API_URL}/tasks/research",
                json={"topic": topic, "num_iterations": 1},
                timeout=30,  # POST just returns task_id (fast) - 30s is plenty
            )
            task_data = resp.json()
            task_id = task_data.get("task_id")

            if task_id:
                st.info(f"🔄 Task started — ID: `{task_id}`")
                result_task = poll_task(task_id)

                if result_task and result_task.get("status") == "completed":
                    result = result_task.get("result", {})
                    st.session_state.last_result = result
                    st.session_state.last_mode = "research"
                    display_research_result(result)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Research complete — Score: {result.get('score', 0):.1f}/100 — PDF: {result.get('file', '')}",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                else:
                    st.error("❌ Research failed or timed out")
                    st.session_state.messages.append({
                        "role": "assistant", "content": "❌ Research failed",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
            else:
                st.error(f"❌ API error: {task_data}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.messages.append({
                "role": "assistant", "content": f"❌ Error: {e}",
                "time": datetime.now().strftime("%H:%M:%S"),
            })

# ============================================================================
# BOTH (RESEARCH + AUDIO)
# ============================================================================

elif both_btn:
    if not topic.strip():
        st.error("❌ Please enter a topic")
    else:
        st.divider()
        st.session_state.messages.append({
            "role": "user",
            "content": f"Research + Audio: {topic}",
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        try:
            resp = requests.post(
                f"{API_URL}/pipeline/run",
                json={"topic": topic},
                timeout=30,  # POST just returns task_id (fast) - 30s is plenty
            )
            task_data = resp.json()
            task_id = task_data.get("task_id")

            if task_id:
                st.info(f"🔄 Pipeline started — ID: `{task_id}`")
                result_task = poll_task(task_id)

                if result_task and result_task.get("status") == "completed":
                    result = result_task.get("result", {})
                    st.session_state.last_result = result
                    st.session_state.last_mode = "both"

                    st.subheader("📄 Research Report")
                    display_research_result(result)

                    st.divider()
                    st.subheader("🎙️ Audio Podcast")
                    display_podcast_result(result)

                    st.divider()
                    st.success("✅ Complete — PDF report and audio podcast are ready!")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Complete — Score: {result.get('score', 0):.1f}/100 — Q&A: {len(result.get('questions', []))} pairs",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                else:
                    st.error("❌ Pipeline failed or timed out")
                    st.session_state.messages.append({
                        "role": "assistant", "content": "❌ Pipeline failed",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
            else:
                st.error(f"❌ API error: {task_data}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.messages.append({
                "role": "assistant", "content": f"❌ Error: {e}",
                "time": datetime.now().strftime("%H:%M:%S"),
            })

# ============================================================================
# SHOW PERSISTED LAST RESULT (on page reload / no button pressed)
# ============================================================================

elif st.session_state.last_result and not clear_btn:
    result = st.session_state.last_result
    mode = st.session_state.last_mode
    st.divider()
    if mode == "research":
        st.subheader("📄 Last Research Report")
        display_research_result(result)
    elif mode == "both":
        st.subheader("📄 Last Research Report")
        display_research_result(result)
        st.divider()
        st.subheader("🎙️ Audio Podcast")
        display_podcast_result(result)

# ============================================================================
# CHAT HISTORY
# ============================================================================

if st.session_state.messages:
    st.divider()
    with st.expander("📋 Session History", expanded=False):
        for msg in st.session_state.messages:
            st.write(f"**{msg['role'].upper()}** ({msg['time']}): {msg['content']}")
