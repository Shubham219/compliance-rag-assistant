"""
Script to create sample regulatory documents for testing.
"""

import os
from pathlib import Path


def create_sample_documents():
    """Create sample regulatory documents"""
    
    sample_docs_path = Path("./data/regulatory_documents")
    sample_docs_path.mkdir(parents=True, exist_ok=True)
    
    # GDPR Sample
    gdpr_content = """
    General Data Protection Regulation (GDPR) - Key Requirements
    
    Article 5: Principles relating to processing of personal data
    1. Personal data shall be processed lawfully, fairly and in a transparent manner.
    2. Collected for specified, explicit and legitimate purposes.
    3. Adequate, relevant and limited to what is necessary (data minimization).
    4. Accurate and kept up to date.
    5. Kept in a form which permits identification for no longer than necessary.
    6. Processed in a manner that ensures appropriate security.
    
    Article 17: Right to erasure ('right to be forgotten')
    The data subject shall have the right to obtain erasure of personal data 
    without undue delay where one of the following grounds applies.
    
    Article 32: Security of processing
    The controller and processor shall implement appropriate technical and 
    organizational measures to ensure a level of security appropriate to the risk.
    """
    
    # SOX Sample
    sox_content = """
    Sarbanes-Oxley Act (SOX) - Key Compliance Requirements
    
    Section 302: Corporate Responsibility for Financial Reports
    - CEO and CFO must certify financial reports
    - Officers responsible for establishing internal controls
    
    Section 404: Management Assessment of Internal Controls
    - Annual report must contain internal control report
    - Management must assess effectiveness of internal controls
    
    Section 802: Criminal Penalties for Altering Documents
    - Fines and/or up to 20 years imprisonment for document destruction
    """
    
    # HIPAA Sample
    hipaa_content = """
    Health Insurance Portability and Accountability Act (HIPAA)
    
    Privacy Rule - Protected Health Information (PHI)
    PHI includes any information that can identify an individual.
    
    Security Rule - Administrative Safeguards
    - Security Management Process
    - Workforce Security
    - Information Access Management
    
    Security Rule - Technical Safeguards
    - Access Control with encryption
    - Audit Controls
    - Transmission Security
    """
    
    # Write files
    (sample_docs_path / "gdpr_requirements.txt").write_text(gdpr_content)
    (sample_docs_path / "sox_compliance.txt").write_text(sox_content)
    (sample_docs_path / "hipaa_security.txt").write_text(hipaa_content)
    
    print(f"✅ Created sample documents in {sample_docs_path}")


if __name__ == "__main__":
    create_sample_documents()