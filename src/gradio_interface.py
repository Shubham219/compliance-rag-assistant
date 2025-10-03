"""
Regulatory Compliance Assistant - Gradio Web UI
================================================
A beautiful, user-friendly web interface for the RAG-based compliance system.

This UI provides:
- Interactive chat interface for compliance queries
- Document upload functionality
- Source citation display
- Query history tracking
- Similarity search
- System statistics dashboard
"""

import gradio as gr
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pandas as pd
from rag_compliance_system import RegulatoryComplianceRAG

# Import the RAG system from the previous code
# Note: In production, you would import from the actual module
# from regulatory_rag import RegulatoryComplianceRAG, create_sample_documents

# For demonstration, we'll assume the RAG class is available
# Include the entire previous code here or import it


class ComplianceUI:
    """
    Web UI wrapper for the Regulatory Compliance RAG system
    
    This class manages the Gradio interface and coordinates between
    user interactions and the underlying RAG system.
    """
    
    def __init__(self):
        """Initialize the UI and RAG system"""
        self.rag_system = None
        self.query_history = []
        self.initialized = False
        
    def initialize_system(
        self,
        llm_provider: str,
        model_name: str,
        api_key: str = None
    ) -> str:
        """
        Initialize the RAG system with user-selected parameters
        
        Args:
            llm_provider: "openai" or "ollama"
            model_name: Specific model to use
            api_key: OpenAI API key (if using OpenAI)
            
        Returns:
            Status message
        """
        try:
            # Set API key if using OpenAI
            if llm_provider == "openai" and api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            elif llm_provider == "openai" and not api_key:
                return "❌ Error: OpenAI API key is required for OpenAI models"
            
            # Initialize RAG system
            self.rag_system = RegulatoryComplianceRAG(
                documents_path="./regulatory_documents",
                vector_db_path="./vector_db",
                llm_provider=llm_provider,
                model_name=model_name,
                temperature=0.1
            )
            
            self.initialized = True
            return "✅ System initialized successfully! You can now start asking compliance questions."
            
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"
    
    def query_compliance(
        self,
        question: str,
        regulation_filter: str,
        show_sources: bool
    ) -> Tuple[str, str, str]:
        """
        Process a compliance query and return formatted results
        
        Args:
            question: User's compliance question
            regulation_filter: Optional regulation type filter
            show_sources: Whether to display source documents
            
        Returns:
            Tuple of (answer, sources_html, status_message)
        """
        if not self.initialized or not self.rag_system:
            return (
                "",
                "",
                "⚠️ Please initialize the system first using the 'System Setup' tab."
            )
        
        if not question.strip():
            return "", "", "⚠️ Please enter a question."
        
        try:
            # Apply regulation filter if selected
            reg_type = None if regulation_filter == "All" else regulation_filter
            
            # Query the RAG system
            result = self.rag_system.query(
                question=question,
                regulation_type=reg_type,
                return_sources=show_sources
            )
            
            # Store in history
            self.query_history.append(result)
            
            # Format answer
            answer = result["answer"]
            
            # Format sources as HTML if requested
            sources_html = ""
            if show_sources and "sources" in result:
                sources_html = self._format_sources_html(result["sources"])
            
            status = f"✅ Query processed successfully at {result['timestamp']}"
            
            return answer, sources_html, status
            
        except Exception as e:
            return "", "", f"❌ Error processing query: {str(e)}"
    
    def _format_sources_html(self, sources: List[Dict]) -> str:
        """
        Format source documents as styled HTML
        
        Args:
            sources: List of source document dictionaries
            
        Returns:
            HTML string with formatted sources
        """
        if not sources:
            return "<p>No sources available.</p>"
        
        html = "<div style='background-color: #f5f5f5; padding: 15px; border-radius: 8px;'>"
        html += "<h3 style='color: #2c3e50; margin-top: 0;'>📚 Source Documents</h3>"
        
        for i, source in enumerate(sources, 1):
            html += f"""
            <div style='background-color: white; padding: 12px; margin: 10px 0; 
                        border-left: 4px solid #3498db; border-radius: 4px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong style='color: #2980b9;'>Source {i}</strong>
                    <span style='background-color: #e8f4f8; padding: 4px 8px; 
                                 border-radius: 4px; font-size: 0.85em;'>
                        Rank: {source['relevance_rank']}
                    </span>
                </div>
                <p style='margin: 10px 0; color: #34495e; line-height: 1.6;'>
                    {source['content'][:300]}...
                </p>
            """
            
            # Add metadata if available
            if source.get('metadata'):
                metadata = source['metadata']
                html += "<div style='font-size: 0.85em; color: #7f8c8d;'>"
                if 'source' in metadata:
                    html += f"📄 <strong>File:</strong> {metadata['source']}<br>"
                if 'page' in metadata:
                    html += f"📖 <strong>Page:</strong> {metadata['page']}"
                html += "</div>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def search_similar_docs(self, query: str, num_results: int) -> Tuple[str, str]:
        """
        Search for similar documents without generating an answer
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Tuple of (results_html, status_message)
        """
        if not self.initialized or not self.rag_system:
            return "", "⚠️ Please initialize the system first."
        
        if not query.strip():
            return "", "⚠️ Please enter a search query."
        
        try:
            results = self.rag_system.get_similar_documents(query, k=num_results)
            
            html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
            html += f"<h3 style='color: #2c3e50;'>🔍 Found {len(results)} similar documents</h3>"
            
            for i, doc in enumerate(results, 1):
                score_color = "#27ae60" if doc['similarity_score'] < 0.5 else "#f39c12"
                
                html += f"""
                <div style='background-color: white; padding: 15px; margin: 10px 0; 
                            border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                        <strong style='color: #2c3e50;'>Result {i}</strong>
                        <span style='background-color: {score_color}; color: white; 
                                     padding: 4px 12px; border-radius: 12px; font-size: 0.9em;'>
                            Similarity: {1 - doc['similarity_score']:.2%}
                        </span>
                    </div>
                    <p style='color: #34495e; line-height: 1.6; margin: 0;'>
                        {doc['content'][:400]}...
                    </p>
                </div>
                """
            
            html += "</div>"
            return html, f"✅ Search completed successfully"
            
        except Exception as e:
            return "", f"❌ Error during search: {str(e)}"
    
    def upload_documents(self, files) -> str:
        """
        Handle document uploads and add to vector store
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            Status message
        """
        if not self.initialized or not self.rag_system:
            return "⚠️ Please initialize the system first."
        
        if not files:
            return "⚠️ No files uploaded."
        
        try:
            from langchain.schema import Document
            
            uploaded_docs = []
            for file in files:
                # Read file content
                if file.name.endswith('.txt'):
                    content = file.read().decode('utf-8')
                    doc = Document(
                        page_content=content,
                        metadata={"source": file.name, "upload_time": datetime.now().isoformat()}
                    )
                    uploaded_docs.append(doc)
                elif file.name.endswith('.pdf'):
                    # For PDFs, save temporarily and use PyPDF loader
                    temp_path = f"./temp_{file.name}"
                    with open(temp_path, 'wb') as f:
                        f.write(file.read())
                    
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    uploaded_docs.extend(docs)
                    
                    # Clean up temp file
                    os.remove(temp_path)
            
            # Add to vector store
            if uploaded_docs:
                self.rag_system.add_documents(uploaded_docs)
                return f"✅ Successfully uploaded and processed {len(uploaded_docs)} document(s)"
            else:
                return "⚠️ No valid documents found in uploaded files"
            
        except Exception as e:
            return f"❌ Error uploading documents: {str(e)}"
    
    def get_query_history(self) -> str:
        """
        Get formatted query history as HTML table
        
        Returns:
            HTML string with query history
        """
        if not self.query_history:
            return "<p>No queries yet. Start asking compliance questions!</p>"
        
        html = """
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>📊 Query History</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>
                    <tr style='background-color: #3498db; color: white;'>
                        <th style='padding: 12px; text-align: left;'>#</th>
                        <th style='padding: 12px; text-align: left;'>Query</th>
                        <th style='padding: 12px; text-align: left;'>Timestamp</th>
                        <th style='padding: 12px; text-align: left;'>Sources</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, query in enumerate(reversed(self.query_history[-10:]), 1):
            bg_color = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            num_sources = query.get('num_sources', 0)
            
            html += f"""
                <tr style='background-color: {bg_color};'>
                    <td style='padding: 10px; border-bottom: 1px solid #ddd;'>{i}</td>
                    <td style='padding: 10px; border-bottom: 1px solid #ddd;'>
                        {query['query'][:60]}...
                    </td>
                    <td style='padding: 10px; border-bottom: 1px solid #ddd; font-size: 0.9em;'>
                        {query['timestamp'][:19]}
                    </td>
                    <td style='padding: 10px; border-bottom: 1px solid #ddd; text-align: center;'>
                        {num_sources}
                    </td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def export_history(self) -> str:
        """
        Export query history to JSON file
        
        Returns:
            Path to exported file
        """
        if not self.query_history:
            return None
        
        filename = f"compliance_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.query_history, f, indent=2)
        
        return filename
    
    def get_system_stats(self) -> str:
        """
        Get system statistics and status
        
        Returns:
            HTML string with system stats
        """
        if not self.initialized or not self.rag_system:
            return "<p>System not initialized.</p>"
        
        # Get vector store stats
        try:
            num_docs = self.rag_system.vector_store.index.ntotal
        except:
            num_docs = "Unknown"
        
        html = f"""
        <div style='background-color: #e8f4f8; padding: 20px; border-radius: 8px;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>📈 System Statistics</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
                <div style='background-color: white; padding: 15px; border-radius: 6px; text-align: center;'>
                    <div style='font-size: 2em; color: #3498db; font-weight: bold;'>{num_docs}</div>
                    <div style='color: #7f8c8d; margin-top: 5px;'>Document Chunks</div>
                </div>
                <div style='background-color: white; padding: 15px; border-radius: 6px; text-align: center;'>
                    <div style='font-size: 2em; color: #27ae60; font-weight: bold;'>{len(self.query_history)}</div>
                    <div style='color: #7f8c8d; margin-top: 5px;'>Total Queries</div>
                </div>
                <div style='background-color: white; padding: 15px; border-radius: 6px; text-align: center;'>
                    <div style='font-size: 1.5em; color: #9b59b6; font-weight: bold;'>
                        {self.rag_system.llm_provider.upper()}
                    </div>
                    <div style='color: #7f8c8d; margin-top: 5px;'>LLM Provider</div>
                </div>
                <div style='background-color: white; padding: 15px; border-radius: 6px; text-align: center;'>
                    <div style='font-size: 1.5em; color: #e74c3c; font-weight: bold;'>
                        {self.rag_system.model_name}
                    </div>
                    <div style='color: #7f8c8d; margin-top: 5px;'>Model</div>
                </div>
            </div>
        </div>
        """
        
        return html


def create_gradio_interface():
    """
    Create and configure the Gradio web interface
    
    This function builds the complete UI with all tabs and components.
    """
    
    # Initialize UI manager
    ui = ComplianceUI()
    
    # Custom CSS for better styling
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
    
    # Create Gradio interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🏛️ Regulatory Compliance Assistant</h1>
                <p>AI-Powered RAG System for Compliance Queries</p>
                <p style="font-size: 0.9em; opacity: 0.9;">
                    GDPR • SOX • HIPAA • and more
                </p>
            </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # TAB 1: System Setup
            with gr.Tab("🚀 System Setup"):
                gr.Markdown("""
                ### Initialize the RAG System
                Configure your AI model and get started. Choose between OpenAI (cloud) or Ollama (local).
                """)
                
                with gr.Row():
                    with gr.Column():
                        llm_provider = gr.Radio(
                            choices=["ollama", "openai"],
                            value="ollama",
                            label="LLM Provider",
                            info="Ollama is free and runs locally. OpenAI requires an API key."
                        )
                        
                        model_name = gr.Textbox(
                            value="llama2",
                            label="Model Name",
                            info="For Ollama: llama2, mistral, etc. For OpenAI: gpt-3.5-turbo, gpt-4"
                        )
                        
                        api_key = gr.Textbox(
                            label="OpenAI API Key (if using OpenAI)",
                            type="password",
                            placeholder="sk-...",
                            info="Only required for OpenAI models"
                        )
                        
                        init_btn = gr.Button("Initialize System", variant="primary", size="lg")
                    
                    with gr.Column():
                        init_status = gr.Textbox(
                            label="Status",
                            lines=10,
                            interactive=False
                        )
                        
                        stats_display = gr.HTML(label="System Stats")
                
                # Connect initialization
                init_btn.click(
                    fn=ui.initialize_system,
                    inputs=[llm_provider, model_name, api_key],
                    outputs=[init_status]
                ).then(
                    fn=ui.get_system_stats,
                    outputs=[stats_display]
                )
            
            # TAB 2: Compliance Q&A
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("""
                ### Ask Compliance Questions
                Get instant answers to your regulatory compliance questions with source citations.
                """)
                
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
                                label="Filter by Regulation",
                                info="Optional: Focus on specific regulation"
                            )
                            
                            show_sources = gr.Checkbox(
                                value=True,
                                label="Show Source Documents",
                                info="Display where the answer came from"
                            )
                        
                        submit_btn = gr.Button("Get Answer", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💡 Example Questions")
                        gr.Examples(
                            examples=[
                                ["What are the key GDPR data protection principles?"],
                                ["What security controls does HIPAA require?"],
                                ["How long do we have to notify about a data breach?"],
                                ["What are SOX 404 requirements for internal controls?"],
                                ["What encryption is required for protected health information?"]
                            ],
                            inputs=question_input
                        )
                
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
                
                sources_output = gr.HTML(label="Source Documents")
                
                status_output = gr.Textbox(label="Status", interactive=False)
                
                # Connect query function
                submit_btn.click(
                    fn=ui.query_compliance,
                    inputs=[question_input, reg_filter, show_sources],
                    outputs=[answer_output, sources_output, status_output]
                )
            
            # TAB 3: Document Search
            with gr.Tab("🔍 Search Documents"):
                gr.Markdown("""
                ### Similarity Search
                Find relevant document sections without generating an answer.
                Useful for exploring what the system knows about a topic.
                """)
                
                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., encryption requirements",
                        lines=2
                    )
                    
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                
                search_btn = gr.Button("Search", variant="primary")
                
                search_results = gr.HTML(label="Search Results")
                search_status = gr.Textbox(label="Status", interactive=False)
                
                # Connect search function
                search_btn.click(
                    fn=ui.search_similar_docs,
                    inputs=[search_query, num_results],
                    outputs=[search_results, search_status]
                )
            
            # TAB 4: Upload Documents
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("""
                ### Add Your Documents
                Upload regulatory documents (PDF or TXT) to expand the knowledge base.
                The system will automatically process and index them.
                """)
                
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt"],
                    type="binary"
                )
                
                upload_btn = gr.Button("Process Documents", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=5)
                
                gr.Markdown("""
                ### 📋 Supported Formats
                - **PDF**: Regulatory texts, compliance guides, audit reports
                - **TXT**: Policy documents, procedures, requirements
                
                **Note**: Large documents will be automatically split into chunks for better retrieval.
                """)
                
                # Connect upload function
                upload_btn.click(
                    fn=ui.upload_documents,
                    inputs=[file_upload],
                    outputs=[upload_status]
                )
            
            # TAB 5: Query History
            with gr.Tab("📊 History & Analytics"):
                gr.Markdown("""
                ### Query History & Analytics
                Track your compliance queries and export for audit purposes.
                """)
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh History", variant="secondary")
                    export_btn = gr.Button("Export to JSON", variant="secondary")
                
                history_display = gr.HTML(label="Recent Queries")
                export_status = gr.Textbox(label="Export Status", interactive=False)
                
                # Connect history functions
                refresh_btn.click(
                    fn=ui.get_query_history,
                    outputs=[history_display]
                )
                
                export_btn.click(
                    fn=ui.export_history,
                    outputs=[export_status]
                )
            
            # TAB 6: Help & Documentation
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                # 📖 User Guide
                
                ## Getting Started
                
                1. **System Setup**: Initialize the system in the first tab
                   - Choose your LLM provider (Ollama for free local, OpenAI for cloud)
                   - Enter your model name
                   - Add API key if using OpenAI
                
                2. **Ask Questions**: Go to the Q&A tab and start asking!
                   - Type your compliance question
                   - Optionally filter by regulation type
                   - Enable source citations to see where answers come from
                
                3. **Upload Documents**: Add your own regulatory documents
                   - Support for PDF and TXT files
                   - Documents are automatically processed and indexed
                   - The more documents you add, the better the answers!
                
                ## Understanding RAG
                
                **RAG (Retrieval Augmented Generation)** works in these steps:
                
                1. 📄 **Load**: Import regulatory documents
                2. ✂️ **Chunk**: Split into manageable pieces
                3. 🔢 **Embed**: Convert text to numerical vectors
                4. 💾 **Store**: Save in vector database (FAISS)
                5. 🔍 **Retrieve**: Find relevant chunks for your query
                6. 🤖 **Generate**: LLM creates answer using retrieved context
                
                ## Tips for Best Results
                
                - ✅ Be specific in your questions
                - ✅ Use the regulation filter when focusing on one standard
                - ✅ Check source citations to verify answers
                - ✅ Add more relevant documents to improve coverage
                - ✅ Use similarity search to explore related content
                
                ## System Requirements
                
                **For Ollama (Local)**:
                - Install from [ollama.ai](https://ollama.ai)
                - Run `ollama pull llama2` or your preferred model
                - No API key needed, runs on your machine
                
                **For OpenAI**:
                - Get API key from [platform.openai.com](https://platform.openai.com)
                - Requires internet connection
                - Pay per use
                
                ## Support
                
                For issues or questions:
                - Check the Status messages for error details
                - Ensure system is initialized before querying
                - Verify document formats (PDF or TXT only)
                - Make sure Ollama is running if using local models
                """)
        
        # Footer
        gr.HTML("""
            <div style='text-align: center; margin-top: 30px; padding: 20px; 
                        background-color: #f8f9fa; border-radius: 8px;'>
                <p style='color: #7f8c8d; margin: 0;'>
                    Built with ❤️ using LangChain, FAISS, and Gradio
                </p>
                <p style='color: #95a5a6; font-size: 0.9em; margin-top: 5px;'>
                    For educational purposes • Always verify compliance information with legal experts
                </p>
            </div>
        """)
    
    return demo


def main():
    """
    Launch the Gradio web interface
    """
    print("="*70)
    print("LAUNCHING REGULATORY COMPLIANCE ASSISTANT WEB UI")
    print("="*70)
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    print("\n✅ Starting web server...")
    print("\n" + "="*70)
    print("🌐 Open your browser and navigate to the URL shown below")
    print("="*70 + "\n")
    
    # Launch with share=True to get a public URL (optional)
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,        # Default Gradio port
        share=True,              # <-- Set to True to create a shareable link
        show_error=True
    )


if __name__ == "__main__":
    main()