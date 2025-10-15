# City Permitting Agent - Streamlit Web Application
import streamlit as st
import pandas as pd
from llama_stack_client import LlamaStackClient, Agent
from llama_stack_client.types import Document
from llama_stack_client import RAGDocument
import uuid
import os
import markdown

# Page configuration
st.set_page_config(
    page_title="City Permitting Agent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .scorecard-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for Llama Stack components
if 'client' not in st.session_state:
    st.session_state.client = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'vector_db_id' not in st.session_state:
    st.session_state.vector_db_id = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_llama_stack():
    """Initialize Llama Stack client and agent"""
    try:
        # Connect to your Llama Stack server
        client = LlamaStackClient(base_url="http://llamastack-server.llama-serve.svc.cluster.local:8321")  # Adapt as needed
        
        # Model: Llama-4-Scout-17B-16E-w4a16 (adjust to your deployment/model)
        model_id = "llama-4-scout-17b-16e-w4a16"
        
        # List of city PDF URLs for ingestion
        permit_pdf_urls = [
            ("http://denvergov.org/content/dam/denvergov/Portals/771/documents/PHI/Food/RevisedFoodRulesandregulationsApril2017compressed.pdf", "application/pdf"),
            ("https://denver.prelive.opencities.com/files/assets/public/v/1/public-health-and-environment/documents/phi/2022_mobileunitguide.pdf", "application/pdf")
        ]
        
        # Create unique vector DB identifier
        vector_db_id = f"v{uuid.uuid4().hex}"
        
        # Register vector DB with an embedding model
        embed_model = next(m for m in client.models.list() if m.model_type == "embedding")
        embedding_model_id = embed_model.identifier
        client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=embedding_model_id,
        )
        
        # Ingest documents into vector DB for RAG search
        documents = [
            RAGDocument(
                document_id=f"permit-doc-{i}",
                content=url,
                mime_type="application/pdf",  # Llama Stack extracts text automatically for PDF
                metadata={"source": "DenverPermitPDF"}
            ) for i, (url, mime_type) in enumerate(permit_pdf_urls)
        ]
        
        client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
        )
        
        # Set up tool groups
        tool_groups = [{
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]}
        }]
        
        # Create the agent with scoring/report instructions
        agent = Agent(
            client,
            model=model_id,
            instructions="You are a city permitting assistant for food trucks and mobile units. Use embedded city codes and health regulations to pre-screen applications, flag errors, detect compliance gaps, and output a detailed scorecard. Summarize gaps and missing sections.",
            tools=tool_groups
        )
        
        # Start session
        session_id = agent.create_session(session_name=f"s-{uuid.uuid4().hex}")
        
        # Store in session state
        st.session_state.client = client
        st.session_state.agent = agent
        st.session_state.session_id = session_id
        st.session_state.vector_db_id = vector_db_id
        st.session_state.initialized = True
        
        return True
    except Exception as e:
        st.error(f"Error initializing Llama Stack: {e}")
        return False

def generate_scorecard(application_form_text):
    """
    Pre-screens the application form, flags errors, missing info, compliance gaps, and returns a scorecard.
    """
    if not st.session_state.agent or not st.session_state.session_id:
        return "Error: Llama Stack agent not initialized. Please check your configuration."
    
    user_prompt = (
        "You are a city permitting AI agent. Using only the embedded city requirements and regulations, "
        "review the following food truck permit application submission for completeness and compliance. "
        "Generate a scorecard listing missing sections, errors, compliance gaps (with reference citations), and a summary compliance score. "
        "Respond in markdown table format for the scorecard, followed by a summary paragraph. \n\n"
        f"Form Content:\n{application_form_text}"
    )

    try:
        # Ask the agent to evaluate the application
        response = st.session_state.agent.create_turn(
            messages=[{"role": "user", "content": user_prompt}],
            session_id=st.session_state.session_id,
            stream=False
        )
        return response.output_message.content
    except Exception as e:
        return f"Error generating scorecard: {str(e)}"

# Main app header
st.markdown("""
<div class="main-header">
    <h1>🏛️ City Permitting Agent</h1>
    <p>AI-Powered Food Truck Permit Application Review</p>
</div>
""", unsafe_allow_html=True)

# Initialize Llama Stack if not already done
if not st.session_state.initialized:
    with st.spinner("Initializing City Permitting Agent... Please wait."):
        if initialize_llama_stack():
            st.success("✅ Llama Stack initialized successfully!")
        else:
            st.error("❌ Failed to initialize Llama Stack. Please check your configuration.")
            st.info("💡 Make sure Llama Stack server is running on http://localhost:8321")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Status indicator
    if st.session_state.initialized:
        st.success("✅ Agent Status: Connected")
    else:
        st.error("❌ Agent Status: Disconnected")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.info("""
    **How it works:**
    1. Fill out your permit application details
    2. Our AI agent reviews against city regulations
    3. Get instant compliance scorecard with recommendations
    4. All decisions reviewed by human officers
    """)
    
    st.divider()
    
    st.header("🏢 Contact Info")
    st.write("**City Permitting Office**")
    st.write("📞 (555) 123-PERMIT")
    st.write("📧 permits@city.gov")

# Main content area
if st.session_state.initialized:
    # Information box
    st.markdown("""
    <div class="info-box">
        <strong>🤖 AI-Powered Review:</strong> Fill out your permit application details below. Our AI agent will review your submission against city regulations and generate a compliance scorecard with recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Form section
    st.header("📋 Permit Application Form")
    
    with st.form("permit_application_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            business_name = st.text_input(
                "🏪 Business Name",
                placeholder="Enter your business name",
                help="Official name of your food truck business"
            )
            
            commissary = st.text_input(
                "🏭 Commissary",
                placeholder="Commissary name and address",
                help="Name and address of your commissary kitchen"
            )
        
        with col2:
            menu = st.text_area(
                "🍽️ Menu Items",
                placeholder="List your menu items (e.g., Tacos, Burritos, Churros)",
                height=100,
                help="List all food items you plan to serve"
            )
            
            additional_info = st.text_area(
                "📄 Additional Information",
                placeholder="Any additional licenses, certifications, or special requirements",
                height=100,
                help="Include any relevant documentation details (optional)"
            )
        
        # Submit button
        submit_button = st.form_submit_button(
            "🔍 Generate Compliance Scorecard",
            type="primary",
            use_container_width=True
        )
    
    # Process form submission
    if submit_button:
        if not business_name or not commissary or not menu:
            st.error("❌ Please fill out all required fields (Business Name, Commissary, Menu Items)")
        else:
            # Construct application form text
            application_form_text = f"""
Business Name: {business_name}
Commissary: {commissary}
Menu: {menu}
Additional Information: {additional_info}
"""
            
            # Generate scorecard
            with st.spinner("🔍 AI Agent reviewing your application... This may take a few moments."):
                scorecard_report = generate_scorecard(application_form_text.strip())
            
            # Display results
            st.header("📊 Compliance Scorecard Results")
            
            # Application summary
            st.markdown(f"""
            <div class="info-box">
                <strong>📋 Application Reviewed:</strong> {business_name}
            </div>
            """, unsafe_allow_html=True)
            
            # Warning about human review
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Important:</strong> This is an AI-generated preliminary review. All applications require final review and approval by a human permitting officer.
            </div>
            """, unsafe_allow_html=True)
            
            # Scorecard results
            st.markdown("""
            <div class="scorecard-container">
                <h4>📊 Compliance Scorecard</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the scorecard
            st.markdown(scorecard_report)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Submit New Application", type="secondary", use_container_width=True):
                    st.rerun()
            
            with col2:
                st.download_button(
                    label="💾 Download Scorecard",
                    data=scorecard_report,
                    file_name=f"permit_scorecard_{business_name.replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col3:
                # Copy to clipboard (simulated with text area)
                if st.button("📋 View Raw Output", type="secondary", use_container_width=True):
                    with st.expander("Raw AI Response (Markdown)", expanded=True):
                        st.text_area(
                            "Raw scorecard output:",
                            value=scorecard_report,
                            height=300,
                            help="You can copy this text for your records"
                        )
            
            # Next steps section
            st.header("🗺️ Next Steps")
            st.markdown("""
            <div class="success-box">
                <strong>📝 Recommended Actions:</strong>
                <ol>
                    <li>Review the compliance scorecard above for any missing requirements or errors</li>
                    <li>Address any compliance gaps identified by the AI agent</li>
                    <li>Gather any missing documentation or certifications</li>
                    <li>Contact the City Permitting Office to schedule a human review appointment</li>
                    <li>Submit your complete application with all required documents</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("❌ City Permitting Agent is not initialized. Please check your Llama Stack configuration.")
    st.info("🔧 Configure your Llama Stack server endpoint and ensure it's running.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🏛️ City Permitting Agent | Powered by Llama Stack & RAG | Human oversight required for all decisions
</div>
""", unsafe_allow_html=True)
