"""
MineMEETS - Production MLOps Multimodal RAG Meeting Intelligence Platform
Gradio-based UI for meeting ingestion and Q&A
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import gradio as gr
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(override=True)

# Import coordinator
from agents.coordinator import MeetingCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize coordinator
coordinator = MeetingCoordinator()

# Data directory
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Helper Functions ====================


def generate_meeting_id() -> str:
    """Generate unique meeting ID."""
    return f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_uploaded_file(file_path: str, file_type: str) -> Tuple[str, str]:
    """Save uploaded file and return meeting_id and saved path."""
    try:
        if not file_path or not os.path.exists(file_path):
            return "", ""

        meeting_id = generate_meeting_id()
        file_extension = Path(file_path).suffix
        dest = RAW_DIR / f"{meeting_id}{file_extension}"

        # Copy file
        import shutil

        shutil.copy(file_path, dest)

        logger.info(f"Saved {file_type} file: {dest}")
        return meeting_id, str(dest)

    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return "", ""


async def process_meeting_async(file_path: str, file_type: str, meeting_id: str) -> Dict[str, Any]:
    """Process meeting file asynchronously."""
    try:
        meeting_data = {
            "id": meeting_id,
            "title": Path(file_path).stem,
            "date": datetime.now().isoformat(),
            "participants": [],
            "content_path": file_path,
            "type": file_type,
        }

        result = await coordinator.process_meeting(meeting_data)
        return result

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {"status": "failed", "error": str(e), "meeting_id": meeting_id}


# ==================== UI Functions ====================


def process_text_file(file) -> str:
    """Process uploaded text file."""
    try:
        if file is None:
            return "❌ No file uploaded"

        meeting_id, saved_path = save_uploaded_file(file.name, "file")
        if not saved_path:
            return "❌ Failed to save file"

        # Run async processing
        result = asyncio.run(process_meeting_async(saved_path, "file", meeting_id))

        if result.get("status") == "processed":
            chunk_count = result.get("chunk_count", result.get("chunks_processed", 0))
            return f"✅ Text processed successfully!\n\nMeeting ID: {meeting_id}\nChunks created: {chunk_count}"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Processing failed: {error}"

    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        return f"❌ Error: {str(e)}"


def process_audio_file(file) -> str:
    """Process uploaded audio file."""
    try:
        if file is None:
            return "❌ No file uploaded"

        meeting_id, saved_path = save_uploaded_file(file.name, "audio")
        if not saved_path:
            return "❌ Failed to save file"

        # Run async processing
        result = asyncio.run(process_meeting_async(saved_path, "audio", meeting_id))

        if result.get("status") == "processed":
            chunk_count = result.get("chunk_count", 0)
            return f"✅ Audio processed successfully!\n\nMeeting ID: {meeting_id}\nChunks created: {chunk_count}\n\n(Audio transcribed via Whisper)"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Processing failed: {error}"

    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return f"❌ Error: {str(e)}"


def process_image_file(file) -> str:
    """Process uploaded image file."""
    try:
        if file is None:
            return "❌ No file uploaded"

        meeting_id, saved_path = save_uploaded_file(file.name, "image")
        if not saved_path:
            return "❌ Failed to save file"

        # Run async processing
        result = asyncio.run(process_meeting_async(saved_path, "image", meeting_id))

        if result.get("status") == "processed":
            return f"✅ Image processed successfully!\n\nMeeting ID: {meeting_id}\n\n(Image embedded via CLIP)"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Processing failed: {error}"

    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return f"❌ Error: {str(e)}"


def build_overlay_instructions(
    assistant_name: str,
    response_style: str,
    preferred_language: str,
    output_template: str,
) -> str:
    """Build assistant overlay instructions for persona + response shape.

    Yes, this is the future "overlay node" payload. AWWW my GAWWD, it's modular.
    """
    return (
        f"assistant_name={assistant_name or 'MineMEETS Assistant'}\n"
        f"response_style={response_style or 'Clear and concise'}\n"
        f"preferred_language={preferred_language or 'English'}\n"
        f"output_template={output_template or 'default'}"
    )


def ask_question(
    meeting_id: str,
    question: str,
    assistant_name: str,
    response_style: str,
    preferred_language: str,
    output_template: str,
) -> str:
    """Ask a question about a meeting."""
    try:
        if not meeting_id or not meeting_id.strip():
            return "❌ Please enter a Meeting ID"

        if not question or not question.strip():
            return "❌ Please enter a question"

        # Run async Q&A
        overlay_instructions = build_overlay_instructions(
            assistant_name,
            response_style,
            preferred_language,
            output_template,
        )
        result = asyncio.run(
            coordinator.qa_agent.process(
                question,
                context={
                    "meeting_id": meeting_id,
                    "overlay_instructions": overlay_instructions,
                },
            )
        )

        if result.get("success"):
            answer = result.get("content", {}).get("answer", "")
            sources = result.get("content", {}).get("sources", [])

            response = f"## Answer\n{answer}\n\n"

            response += "### Assistant Overlay\n"
            response += f"- Name: {assistant_name or 'MineMEETS Assistant'}\n"
            response += f"- Style: {response_style or 'Clear and concise'}\n"
            response += f"- Language: {preferred_language or 'English'}\n"

            if sources:
                response += f"**Sources:** {len(sources)} contexts retrieved\n"

            return response
        else:
            return f"❌ Q&A failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        logger.error(f"Q&A error: {str(e)}")
        return f"❌ Error: {str(e)}"


def get_meeting_summary(meeting_id: str) -> str:
    """Get comprehensive meeting summary."""
    try:
        if not meeting_id or not meeting_id.strip():
            return "❌ Please enter a Meeting ID"

        # Run async summary generation
        result = asyncio.run(coordinator.qa_agent.multimodal_rag.get_meeting_summary(meeting_id))

        if result.get("success"):
            summary = result.get("summary", "")
            modalities = result.get("modalities", [])
            stats = result.get("stats", {})

            response = f"**Meeting Summary:**\n\n{summary}\n\n"
            response += f"**Modalities Used:** {', '.join(modalities)}\n"
            response += f"**Total Chunks:** {stats.get('total_chunks', 0)}\n"

            return response
        else:
            return f"❌ Summary generation failed: {result.get('summary', 'Unknown error')}"

    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        return f"❌ Error: {str(e)}"


def get_database_stats() -> str:
    """Get database statistics."""
    try:
        stats = coordinator.pinecone_db.get_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        namespaces = stats.get("namespaces", {})

        response = f"**Database Statistics:**\n\n"
        response += f"Total Vectors: {total_vectors}\n"
        response += f"Total Meetings: {len(namespaces)}\n\n"

        if namespaces:
            response += "**Meetings:**\n"
            for ns, ns_stats in list(namespaces.items())[:10]:  # Show first 10
                response += f"- {ns}: {ns_stats.get('vector_count', 0)} vectors\n"

        return response

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return f"❌ Error getting stats: {str(e)}"


# ==================== Gradio Interface ====================


def create_interface():
    """Create Gradio interface."""

    custom_css = """
    .hero-card {border-radius: 14px; padding: 14px; background: linear-gradient(135deg,#15192b,#232b4a); color: #f3f5ff;}
    .stat-chip {display:inline-block; padding:6px 10px; border-radius: 20px; background:#f0f4ff; margin: 4px;}
    """

    with gr.Blocks(
        title="MineMEETS - MLOps Meeting Intelligence",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:
        gr.Markdown(
            """
        <div class="hero-card">
        <h1>🎯 MineMEETS — Multimodal RAG Meeting Intelligence</h1>
        <p><b>Production MLOps Platform</b> for processing and querying meeting content across text, audio, and images.</p>
        <p>"Use the Force, but verify with sources." — definitely from the Director's Cut.</p>
        </div>
        
        ### How to Use:
        1. **Upload** your meeting files (text transcripts, audio recordings, or images)
        2. **Process** each file to extract and embed content
        3. **Copy** the Meeting ID from the success message
        4. **Ask questions** using the Meeting ID in the Q&A tab
        """
        )

        with gr.Tabs():
            # ========== Upload & Process Tab ==========
            with gr.Tab("📁 Upload & Process"):
                gr.Markdown("### Upload Meeting Content")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📝 Text Files (.txt, .pdf, .docx)")
                        text_file = gr.File(
                            label="Upload Text File", file_types=[".txt", ".pdf", ".docx"]
                        )
                        text_btn = gr.Button("Process Text", variant="primary")
                        text_output = gr.Textbox(label="Processing Result", lines=5)
                        text_btn.click(process_text_file, inputs=[text_file], outputs=[text_output])

                    with gr.Column():
                        gr.Markdown("#### 🎙️ Audio Files (.mp3, .wav, .m4a)")
                        audio_file = gr.File(
                            label="Upload Audio File", file_types=[".mp3", ".wav", ".m4a", ".ogg"]
                        )
                        audio_btn = gr.Button("Process Audio", variant="primary")
                        audio_output = gr.Textbox(label="Processing Result", lines=5)
                        audio_btn.click(
                            process_audio_file, inputs=[audio_file], outputs=[audio_output]
                        )

                    with gr.Column():
                        gr.Markdown("#### 🖼️ Image Files (.png, .jpg, .jpeg)")
                        image_file = gr.File(
                            label="Upload Image File", file_types=[".png", ".jpg", ".jpeg", ".webp"]
                        )
                        image_btn = gr.Button("Process Image", variant="primary")
                        image_output = gr.Textbox(label="Processing Result", lines=5)
                        image_btn.click(
                            process_image_file, inputs=[image_file], outputs=[image_output]
                        )

            # ========== Q&A Tab ==========
            with gr.Tab("💬 Q&A"):
                gr.Markdown("### Ask Questions About Your Meetings")

                with gr.Row():
                    meeting_id_input = gr.Textbox(
                        label="Meeting ID",
                        placeholder="e.g., meeting_20260131_143022",
                        info="Enter the Meeting ID from the processing step",
                    )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What were the main discussion points?",
                        lines=3,
                    )

                gr.Markdown("### 🤖 Assistant Overlay (PA-ready nodes)")
                with gr.Row():
                    assistant_name_input = gr.Textbox(
                        label="Assistant Name",
                        value="MineMEETS Assistant",
                        info="PA overlay node: identity",
                    )
                    preferred_language_input = gr.Textbox(
                        label="Preferred Language",
                        value="English",
                        info="PA overlay node: language",
                    )

                with gr.Row():
                    response_style_input = gr.Dropdown(
                        label="Response Style",
                        choices=[
                            "Clear and concise",
                            "Executive summary",
                            "Action-items first",
                            "Detailed analyst mode",
                        ],
                        value="Clear and concise",
                    )
                    output_template_input = gr.Textbox(
                        label="Output Template",
                        placeholder="e.g. summary|decisions|risks|next_steps",
                        info="PA overlay node: output schema",
                    )

                with gr.Row():
                    ask_btn = gr.Button("Ask Question", variant="primary", size="lg")
                    summary_btn = gr.Button("Get Meeting Summary", variant="secondary")

                answer_output = gr.Markdown(label="Answer")

                ask_btn.click(
                    ask_question,
                    inputs=[
                        meeting_id_input,
                        question_input,
                        assistant_name_input,
                        response_style_input,
                        preferred_language_input,
                        output_template_input,
                    ],
                    outputs=[answer_output],
                )

                summary_btn.click(
                    get_meeting_summary, inputs=[meeting_id_input], outputs=[answer_output]
                )

                gr.Markdown(
                    """
                ### Example Questions:
                - What were the main topics discussed?
                - What decisions were made?
                - Who spoke about [topic]?
                - What action items were identified?
                """
                )

            # ========== Database Stats Tab ==========
            with gr.Tab("📊 Database Stats"):
                gr.Markdown("### Vector Database Statistics")

                stats_output = gr.Markdown()
                stats_btn = gr.Button("Refresh Stats", variant="primary")

                stats_btn.click(get_database_stats, outputs=[stats_output])

                # Load stats on startup
                app.load(get_database_stats, outputs=[stats_output])

        gr.Markdown(
            """
        ---
        **MineMEETS** — Built with MLOps best practices | [GitHub](https://github.com/yourusername/MineMEETS)
        """
        )

    return app


# ==================== Main ====================

if __name__ == "__main__":
    logger.info("Starting MineMEETS Gradio application...")

    # Check environment
    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set in environment!")

    if not os.getenv("OLLAMA_HOST"):
        logger.warning("OLLAMA_HOST not set, using default: http://localhost:11434")

    # Create and launch interface
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True,
    )
