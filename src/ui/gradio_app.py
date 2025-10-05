"""
Gradio Web Interface for Compliance RAG Assistant.
"""

import gradio as gr
from typing import Tuple, List
from pathlib import Path

from ..core.rag_engine import RegulatoryComplianceRAG
from ..utils.logger import app_logger
from ..utils.config import config
from .components import ComplianceUI


def create_gradio_interface():
    """Create and configure the Gradio web interface"""
    
    # Initialize UI manager
    ui = ComplianceUI()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Compliance RAG Assistant") as demo:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🏛️ Regulatory Compliance Assistant</h1>
                <p>AI-Powered RAG System for Compliance Queries</p>
                <p style="font-size: 0.9em; opacity: 0.9;">
                    FATF • GDPR • SOX • HIPAA • PCI-DSS • ISO 27001
                </p>
            </div>
        """)
        
        with gr.Tabs():
            
            # TAB 1: System Setup
            with gr.Tab("🚀 System Setup"):
                gr.Markdown("### Initialize the RAG System")
                
                with gr.Row():
                    with gr.Column():
                        llm_provider = gr.Radio(
                            choices=["ollama", "openai"],
                            value="ollama",
                            label="LLM Provider"
                        )
                        model_name = gr.Textbox(
                            value="llama2",
                            label="Model Name"
                        )
                        api_key = gr.Textbox(
                            label="OpenAI API Key (if using OpenAI)",
                            type="password",
                            placeholder="sk-..."
                        )
                        init_btn = gr.Button("Initialize System", variant="primary", size="lg")
                    
                    with gr.Column():
                        init_status = gr.Textbox(label="Status", lines=10, interactive=False)
                        stats_display = gr.HTML(label="System Stats")
                
                init_btn.click(
                    fn=ui.initialize_system,
                    inputs=[llm_provider, model_name, api_key],
                    outputs=[init_status]
                ).then(
                    fn=ui.get_system_stats,
                    outputs=[stats_display]
                )
            
            # TAB 2: Q&A
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("### Ask Compliance Questions")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Your Compliance Question",
                            placeholder="e.g., What are the GDPR requirements for data retention?",
                            lines=3
                        )
                        
                        with gr.Row():
                            reg_filter = gr.Dropdown(
                                choices=["All", "GDPR", "SOX", "HIPAA", "PCI-DSS", "ISO 27001"],
                                value="All",
                                label="Filter by Regulation"
                            )
                            show_sources = gr.Checkbox(
                                value=True,
                                label="Show Sources"
                            )
                        
                        submit_btn = gr.Button("Get Answer", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💡 Example Questions")
                        gr.Examples(
                            examples=[
                                ["What are the key GDPR data protection principles?"],
                                ["What security controls does HIPAA require?"],
                                ["How long do we have to notify about a data breach?"],
                                ["What are SOX 404 requirements?"],
                            ],
                            inputs=question_input
                        )
                
                answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
                sources_output = gr.HTML(label="Source Documents")
                status_output = gr.Textbox(label="Status", interactive=False)
                
                submit_btn.click(
                    fn=ui.query_compliance,
                    inputs=[question_input, reg_filter, show_sources],
                    outputs=[answer_output, sources_output, status_output]
                )
            
            # TAB 3: Document Search
            with gr.Tab("🔍 Search Documents"):
                gr.Markdown("### Similarity Search")
                
                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., encryption requirements",
                        lines=2
                    )
                    num_results = gr.Slider(1, 10, value=5, step=1, label="Results")
                
                search_btn = gr.Button("Search", variant="primary")
                search_results = gr.HTML(label="Results")
                search_status = gr.Textbox(label="Status", interactive=False)
                
                search_btn.click(
                    fn=ui.search_similar_docs,
                    inputs=[search_query, num_results],
                    outputs=[search_results, search_status]
                )
            
            # TAB 4: Upload Documents
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("### Add Your Documents")
                
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx"]
                )
                upload_btn = gr.Button("Process Documents", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False, lines=5)
                
                upload_btn.click(
                    fn=ui.upload_documents,
                    inputs=[file_upload],
                    outputs=[upload_status]
                )
            
            # TAB 5: History
            with gr.Tab("📊 History"):
                gr.Markdown("### Query History")
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    export_btn = gr.Button("Export", variant="secondary")
                
                history_display = gr.HTML(label="Recent Queries")
                export_status = gr.Textbox(label="Export Status", interactive=False)
                
                refresh_btn.click(fn=ui.get_query_history, outputs=[history_display])
                export_btn.click(fn=ui.export_history, outputs=[export_status])
        
        # Footer
        gr.HTML("""
            <div style='text-align: center; margin-top: 30px; padding: 20px; 
                        background-color: #f8f9fa; border-radius: 8px;'>
                <p style='color: #7f8c8d; margin: 0;'>
                    Built with ❤️ using LangChain, FAISS, and Gradio
                </p>
            </div>
        """)
    
    return demo


def main():
    """Launch the Gradio interface"""
    app_logger.info("Starting Gradio Web Interface...")
    
    demo = create_gradio_interface()
    
    demo.launch(
        server_name=config.get("ui.host", "0.0.0.0"),
        server_port=config.get("ui.port", 7860),
        share=config.get("ui.share", False),
        show_error=True
    )


if __name__ == "__main__":
    main()
