"""
UI Components and logic for Gradio interface.
"""

import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
from pathlib import Path

from ..core.rag_engine import RegulatoryComplianceRAG
from ..utils.logger import app_logger


class ComplianceUI:
    """Manages UI state and interactions"""
    
    def __init__(self):
        self.rag_system = None
        self.query_history = []
        self.initialized = False
    
    def initialize_system(
        self,
        llm_provider: str,
        model_name: str,
        api_key: str = None
    ) -> str:
        """Initialize the RAG system"""
        try:
            if llm_provider == "openai" and api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            elif llm_provider == "openai" and not api_key:
                return "❌ Error: OpenAI API key required"
            
            self.rag_system = RegulatoryComplianceRAG(
                llm_provider=llm_provider,
                model_name=model_name,
                auto_load=True
            )
            
            self.initialized = True
            return "✅ System initialized successfully!"
            
        except Exception as e:
            app_logger.error(f"Initialization error: {e}")
            return f"❌ Error: {str(e)}"
    
    def query_compliance(
        self,
        question: str,
        regulation_filter: str,
        show_sources: bool
    ) -> Tuple[str, str, str]:
        """Process compliance query"""
        if not self.initialized:
            return "", "", "⚠️ Please initialize the system first"
        
        if not question.strip():
            return "", "", "⚠️ Please enter a question"
        
        try:
            reg_type = None if regulation_filter == "All" else regulation_filter
            
            result = self.rag_system.query(
                question=question,
                regulation_type=reg_type,
                return_sources=show_sources
            )
            
            answer = result["answer"]
            sources_html = ""
            
            if show_sources and "sources" in result:
                sources_html = self._format_sources_html(result["sources"])
            
            status = f"✅ Query processed at {result['timestamp']}"
            return answer, sources_html, status
            
        except Exception as e:
            app_logger.error(f"Query error: {e}")
            return "", "", f"❌ Error: {str(e)}"
    
    def _format_sources_html(self, sources: List[Dict]) -> str:
        """Format source documents as HTML"""
        if not sources:
            return "<p>No sources available.</p>"
        
        html = "<div style='background-color: #f5f5f5; padding: 15px; border-radius: 8px;'>"
        html += "<h3>📚 Source Documents</h3>"
        
        for i, source in enumerate(sources, 1):
            html += f"""
            <div style='background-color: white; padding: 12px; margin: 10px 0; 
                        border-left: 4px solid #3498db; border-radius: 4px;'>
                <strong>Source {i}</strong> (Rank: {source['relevance_rank']})
                <p>{source['content'][:300]}...</p>
            </div>
            """
        
        html += "</div>"
        return html
    
    def search_similar_docs(self, query: str, num_results: int) -> Tuple[str, str]:
        """Search similar documents"""
        if not self.initialized:
            return "", "⚠️ System not initialized"
        
        if not query.strip():
            return "", "⚠️ Enter a search query"
        
        try:
            results = self.rag_system.search_similar_documents(query, k=num_results)
            
            html = f"<h3>🔍 Found {len(results)} similar documents</h3>"
            
            for i, doc in enumerate(results, 1):
                score_color = "#27ae60" if doc['similarity_score'] < 0.5 else "#f39c12"
                
                html += f"""
                <div style='background-color: white; padding: 15px; margin: 10px 0; 
                            border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <strong>Result {i}</strong>
                    <span style='background-color: {score_color}; color: white; 
                                 padding: 4px 12px; border-radius: 12px;'>
                        Similarity: {1 - doc['similarity_score']:.2%}
                    </span>
                    <p>{doc['content'][:400]}...</p>
                </div>
                """
            
            return html, "✅ Search completed"
            
        except Exception as e:
            return "", f"❌ Error: {str(e)}"
    
    def upload_documents(self, files) -> str:
        """Handle document uploads"""
        if not self.initialized:
            return "⚠️ Initialize system first"
        
        if not files:
            return "⚠️ No files uploaded"
        
        try:
            file_paths = []
            for file in files:
                # Save temporarily
                temp_path = f"./temp_{file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(file.read() if hasattr(file, 'read') else file)
                file_paths.append(temp_path)
            
            # Add to system
            success = self.rag_system.add_documents(file_paths)
            
            # Cleanup
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            if success:
                return f"✅ Successfully processed {len(files)} file(s)"
            else:
                return "❌ Error processing documents"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_query_history(self) -> str:
        """Get query history as HTML"""
        if not self.initialized or not self.rag_system:
            return "<p>No history available</p>"
        
        history = self.rag_system.query_history
        
        if not history:
            return "<p>No queries yet</p>"
        
        html = "<h3>📊 Recent Queries</h3><table style='width: 100%;'>"
        html += "<tr><th>#</th><th>Query</th><th>Time</th></tr>"
        
        for i, query in enumerate(reversed(history[-10:]), 1):
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{query['query'][:60]}...</td>
                <td>{query['timestamp'][:19]}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def export_history(self) -> str:
        """Export query history"""
        if not self.initialized:
            return "⚠️ System not initialized"
        
        try:
            filename = f"compliance_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.rag_system.export_query_history(filename)
            return f"✅ Exported to {filename}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_system_stats(self) -> str:
        """Get system statistics as HTML"""
        if not self.initialized:
            return "<p>System not initialized</p>"
        
        stats = self.rag_system.get_system_stats()
        
        html = f"""
        <div style='background-color: #e8f4f8; padding: 20px; border-radius: 8px;'>
            <h3>📈 System Statistics</h3>
            <p><strong>LLM Provider:</strong> {stats['llm_provider']}</p>
            <p><strong>Model:</strong> {stats['model_name']}</p>
            <p><strong>Queries:</strong> {stats['num_queries']}</p>
            <p><strong>Vectors:</strong> {stats['vector_store'].get('num_vectors', 0)}</p>
        </div>
        """
        return html
