# City Permitting AI Agent - Streamlit Application
import streamlit as st
import json
import os
import requests
import tempfile
import uuid
from typing import Dict, Any, List
from io import BytesIO

# Llama Stack imports
try:
    from llama_stack_client import LlamaStackClient
    from llama_stack_client.types import Document
except ImportError:
    st.error("Please install llama-stack-client: pip install llama-stack-client")
    st.stop()

# PDF processing
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Please install pypdf: pip install pypdf")
    st.stop()

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for the City Permitting Agent"""
    
    # Llama Stack Configuration
    LLAMA_STACK_HOST = os.getenv("LLAMA_STACK_HOST", "localhost")
    LLAMA_STACK_PORT = os.getenv("LLAMA_STACK_PORT", "8321")
    #LLAMA_STACK_URL = f"http://{LLAMA_STACK_HOST}:{LLAMA_STACK_PORT}"
    LLAMA_STACK_URL = "http://llamastack-server.llama-serve.svc.cluster.local:8321"
    
    # Model Configuration
    MODEL_ID = "llama-4-scout-17b-16e-w4a16"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Denver Permit Document URLs with fallbacks
    PERMIT_DOCS = {
        "food_rules_2017": {
            "urls": [
                # Primary URL
                "https://www.denvergov.org/files/assets/public/public-health-and-environment/documents/phi/food/revisedfoodrulesandregulationsapril2017compressed.pdf",
                # Fallback URLs
                "http://denvergov.org/content/dam/denvergov/Portals/771/documents/PHI/Food/RevisedFoodRulesandregulationsApril2017compressed.pdf",
            ],
            "description": "Denver Food Rules and Regulations April 2017"
        },
        "mobile_unit_guide_2022": {
            "urls": [
                "https://denver.prelive.opencities.com/files/assets/public/v/1/public-health-and-environment/documents/phi/2022_mobileunitguide.pdf",
            ],
            "description": "Denver Mobile Unit Guide 2022"
        }
    }
    
    # Fallback content if PDFs cannot be downloaded
    FALLBACK_CONTENT = """
    DENVER MOBILE FOOD TRUCK PERMIT REQUIREMENTS
    
    LICENSE REQUIREMENTS:
    - City and County of Denver 'Retail Food Establishment-Mobile' license required
    - Complete Mobile Plan Review Packet submission
    - Processing time: 30 days during busy season
    - Annual renewal required
    
    WATER SYSTEM REQUIREMENTS:
    - Hand washing sink: minimum 10 inches wide x 10 inches long x 5 inches deep
    - Water temperature: 100°F to 120°F at the faucet
    - Soap and single-use paper towels required at all times
    - Minimum 10 gallons clean water tank OR 3 gallons per hour of operation (whichever is greater)
    - Wastewater tank must be at least 15% larger than clean water tank
    - All water tanks must be NSF-approved and labeled
    
    COMMISSARY REQUIREMENTS:
    - Must operate from an approved commissary facility
    - Report to commissary daily for food preparation, cleaning, and servicing
    - Affidavit of Commissary form required
    - Commissary must be licensed by Denver or approved jurisdiction
    
    LOCATION RESTRICTIONS:
    - Cannot operate in Central Business District without special permit
    - 300 feet minimum from public parks (unless during special event with permission)
    - 200 feet minimum from other food trucks
    - 200 feet minimum from eating/drinking establishments (unless written consent)
    - 50 feet minimum from residential zoning districts
    - Cannot block fire hydrants, crosswalks, or handicap access
    
    EQUIPMENT REQUIREMENTS:
    - Fire suppression system required for equipment producing grease-laden vapors
    - Type I hood system required for grills, fryers, etc.
    - Commercial-grade equipment only (no residential appliances)
    - All equipment must be NSF or equivalent certified
    - Adequate ventilation system required
    
    FOOD SAFETY REQUIREMENTS:
    - All food stored minimum 6 inches above ground
    - Cold potentially hazardous food: 41°F or below
    - Hot potentially hazardous food: 135°F or above
    - Accurate thermometers required (± 2°F accuracy)
    - Food protection from contamination at all times
    - No bare hand contact with ready-to-eat foods
    
    STRUCTURAL REQUIREMENTS:
    - Floors: smooth, non-absorbent, easily cleanable
    - Walls and ceilings: light-colored, smooth, easily cleanable
    - Adequate lighting: minimum 10 foot-candles on food prep surfaces
    - Sneeze guards required for customer self-service
    - Waste containers with lids required
    
    DOCUMENTATION REQUIRED FOR PERMIT:
    1. Completed application form with fees
    2. Vehicle registration and proof of ownership
    3. Insurance certificate (general liability)
    4. Commissary affidavit (signed by commissary owner)
    5. Mobile unit floor plan (to scale)
    6. Equipment specification sheets
    7. Menu list
    8. Water system diagram
    9. Waste disposal plan
    10. Certified food manager certificate (at least one per unit)
    
    INSPECTION REQUIREMENTS:
    - Initial inspection required before permit issuance
    - Routine unannounced inspections throughout operation
    - Must maintain score of 80 or above
    - Critical violations must be corrected immediately
    - Re-inspection fee applies for follow-up inspections
    
    FEES (Subject to change):
    - New mobile unit application: varies by unit type
    - Annual renewal: varies by unit type
    - Re-inspection fee: if applicable
    - Late renewal penalty: if applicable
    """

# ============================================================================
# Document Loader with Robust Error Handling
# ============================================================================

class DocumentLoader:
    """Handles document loading with fallback mechanisms"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def download_pdf(self, urls: List[str], description: str) -> bytes:
        """Download PDF with multiple fallback URLs"""
        for url in urls:
            try:
                st.info(f"Attempting to download: {description}")
                response = self.session.get(
                    url,
                    allow_redirects=True,
                    timeout=30,
                    verify=True
                )
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'pdf' in content_type.lower() or len(response.content) > 1000:
                        st.success(f"✓ Downloaded: {description}")
                        return response.content
                    else:
                        st.warning(f"Response not PDF format from {url}")
                else:
                    st.warning(f"Status {response.status_code} from {url}")
                    
            except requests.exceptions.RequestException as e:
                st.warning(f"Failed to download from {url}: {str(e)}")
                continue
            except Exception as e:
                st.warning(f"Unexpected error with {url}: {str(e)}")
                continue
        
        return None
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def load_permit_documents(self) -> List[Document]:
        """Load all permit requirement documents"""
        documents = []
        
        for doc_id, doc_info in Config.PERMIT_DOCS.items():
            # Try to download PDF
            pdf_content = self.download_pdf(doc_info["urls"], doc_info["description"])
            
            if pdf_content:
                # Extract text
                text_content = self.extract_text_from_pdf(pdf_content)
                
                if text_content and len(text_content.strip()) > 100:
                    doc = Document(
                        document_id=doc_id,
                        content=text_content,
                        mime_type="text/plain",
                        metadata={
                            "source": doc_info["urls"][0],
                            "description": doc_info["description"],
                            "type": "permit_requirements"
                        }
                    )
                    documents.append(doc)
                else:
                    st.warning(f"Insufficient content extracted from {doc_info['description']}")
        
        # If no documents loaded, use fallback content
        if not documents:
            st.warning("⚠�? Could not download PDFs. Using fallback permit requirements.")
            fallback_doc = Document(
                document_id="fallback_requirements",
                content=Config.FALLBACK_CONTENT,
                mime_type="text/plain",
                metadata={
                    "source": "fallback",
                    "description": "Denver Permit Requirements (Fallback)",
                    "type": "permit_requirements"
                }
            )
            documents.append(fallback_doc)
        
        return documents

# ============================================================================
# Llama Stack Agent Manager
# ============================================================================

class PermitAgentManager:
    """Manages Llama Stack client and agent operations"""
    
    def __init__(self):
        self.client = None
        self.vector_db_id = None
        self.session_id = None
        self.messages = []
    
    def initialize_client(self) -> bool:
        """Initialize Llama Stack client"""
        try:
            self.client = LlamaStackClient(base_url=Config.LLAMA_STACK_URL)
            # Test connection
            self.client.models.list()
            return True
        except Exception as e:
            st.error(f"Failed to connect to Llama Stack at {Config.LLAMA_STACK_URL}: {str(e)}")
            return False
    
    def setup_vector_db(self, documents: List[Document]) -> bool:
        """Setup vector database and ingest documents"""
        try:
            # Generate unique vector DB ID
            self.vector_db_id = f"permit-db-{uuid.uuid4().hex[:8]}"
            
            # Get available providers
            providers = self.client.providers.list()
            vector_provider = None
            
            for provider in providers:
                if hasattr(provider, 'api') and provider.api == 'vector_io':
                    vector_provider = provider
                    break
            
            if not vector_provider:
                st.error("No vector_io provider found in Llama Stack")
                return False
            
            # Register vector database
            self.client.vector_dbs.register(
                vector_db_id=self.vector_db_id,
                provider_id=vector_provider.provider_id,
                embedding_model=Config.EMBEDDING_MODEL,
                embedding_dimension=Config.EMBEDDING_DIMENSION
            )
            
            # Ingest documents
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=1024
            )
            
            st.success(f"✓ Vector database setup complete: {self.vector_db_id}")
            return True
            
        except Exception as e:
            st.error(f"Error setting up vector database: {str(e)}")
            return False
    
    def create_session(self):
        """Create new conversation session"""
        self.session_id = f"session-{uuid.uuid4().hex[:8]}"
        self.messages = [
            {
                "role": "system",
                "content": """You are an expert City Permitting AI Agent for Denver food truck permits.

Your responsibilities:
1. Review permit applications for completeness and accuracy
2. Check compliance with Denver food truck regulations
3. Identify missing information or errors
4. Provide clear, actionable feedback with specific regulation references
5. Generate evaluation scorecards with scores from 0-100

Always be professional, thorough, and cite specific regulations when providing feedback."""
            }
        ]
    
    def query_with_rag(self, query: str) -> str:
        """Query with RAG context"""
        try:
            # Query vector database for relevant context
            rag_results = self.client.tool_runtime.rag_tool.query(
                content=query,
                vector_db_ids=[self.vector_db_id]
            )
            
            # Extract context from RAG results
            rag_context = []
            if hasattr(rag_results, 'content') and rag_results.content:
                for chunk in rag_results.content:
                    if hasattr(chunk, 'text'):
                        rag_context.append(chunk.text)
            
            # Build enhanced prompt with RAG context
            enhanced_query = query
            if rag_context:
                context_text = "\n\n".join(rag_context)
                enhanced_query = f"""{query}

RELEVANT DENVER REGULATIONS:
{context_text}

Base your response on the regulations provided above."""
            
            # Add to conversation messages
            self.messages.append({
                "role": "user",
                "content": enhanced_query
            })
            
            # Get LLM response using Responses API (chat completion)
            response = self.client.inference.chat_completion(
                model_id=Config.MODEL_ID,
                messages=self.messages
            )
            
            # Extract response content
            if hasattr(response, 'completion_message'):
                response_text = response.completion_message.content
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
            
            # Add assistant response to messages
            self.messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error querying agent: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def evaluate_application(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate permit application"""
        
        evaluation_query = f"""Evaluate this Denver food truck permit application:

APPLICATION DATA:
{json.dumps(application, indent=2)}

Provide a detailed evaluation in JSON format with:
{{
  "overall_score": <0-100>,
  "recommendation": "APPROVED" | "NEEDS_REVISION" | "REJECTED",
  "categories": {{
    "completeness": {{"score": <0-100>, "findings": [...], "required_actions": [...]}},
    "accuracy": {{"score": <0-100>, "findings": [...], "required_actions": [...]}},
    "compliance": {{"score": <0-100>, "findings": [...], "required_actions": [...]}},
    "documentation": {{"score": <0-100>, "findings": [...], "required_actions": [...]}},
    "safety_requirements": {{"score": <0-100>, "findings": [...], "required_actions": [...]}}
  }},
  "summary": "<brief summary>",
  "next_steps": [...]
}}"""
        
        response = self.query_with_rag(evaluation_query)
        
        # Try to parse JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                # Fallback structure
                evaluation = {
                    "overall_score": 0,
                    "recommendation": "NEEDS_REVIEW",
                    "raw_response": response
                }
        except:
            evaluation = {
                "overall_score": 0,
                "recommendation": "ERROR",
                "raw_response": response
            }
        
        return evaluation

# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Denver Food Truck Permit Assistant",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = None
        st.session_state.initialized = False
        st.session_state.documents_loaded = False
    
    if "evaluation_history" not in st.session_state:
        st.session_state.evaluation_history = []
    
    # Sidebar
    with st.sidebar:
        st.title("🚚 Denver Permit Assistant")
        st.markdown("---")
        
        # Configuration
        st.subheader("Configuration")
        llama_host = st.text_input("Llama Stack Host", value=Config.LLAMA_STACK_HOST)
        llama_port = st.text_input("Llama Stack Port", value=Config.LLAMA_STACK_PORT)
        
        Config.LLAMA_STACK_HOST = llama_host
        Config.LLAMA_STACK_PORT = llama_port
        Config.LLAMA_STACK_URL = f"http://{llama_host}:{llama_port}"
        
        st.markdown("---")
        
        # Initialization
        if st.button("🚀 Initialize Agent", type="primary"):
            with st.spinner("Initializing agent..."):
                try:
                    # Create agent manager
                    agent = PermitAgentManager()
                    
                    # Step 1: Connect to Llama Stack
                    st.info("Connecting to Llama Stack...")
                    if not agent.initialize_client():
                        st.error("Failed to initialize Llama Stack client")
                        st.stop()
                    
                    # Step 2: Load documents
                    st.info("Loading permit documents...")
                    loader = DocumentLoader()
                    documents = loader.load_permit_documents()
                    
                    if not documents:
                        st.error("No documents loaded")
                        st.stop()
                    
                    st.success(f"Loaded {len(documents)} document(s)")
                    
                    # Step 3: Setup vector database
                    st.info("Setting up vector database...")
                    if not agent.setup_vector_db(documents):
                        st.error("Failed to setup vector database")
                        st.stop()
                    
                    # Step 4: Create session
                    agent.create_session()
                    
                    # Save to session state
                    st.session_state.agent = agent
                    st.session_state.initialized = True
                    st.session_state.documents_loaded = True
                    
                    st.success("✓ Agent initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Initialization error: {str(e)}")
                    st.exception(e)
        
        # Status
        st.markdown("---")
        st.subheader("Status")
        if st.session_state.initialized:
            st.success("✓ Agent Ready")
        else:
            st.warning("⚠�? Agent Not Initialized")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        **Denver Food Truck Permit AI Agent**
        
        Features:
        - �? Automated compliance checking
        - ✅ Completeness verification
        - 📊 Scorecard evaluation
        - 📚 Denver regulations knowledge base
        - 🤖 RAG-powered responses
        
        Built with:
        - Llama Stack
        - Llama 4 Scout 17B
        - Model Context Protocol (MCP)
        - Retrieval Augmented Generation (RAG)
        """)
    
    # Main content
    st.title("🚚 Denver Food Truck Permit Application Review")
    st.markdown("AI-powered permit application evaluation system")
    
    # Check if agent is initialized
    if not st.session_state.initialized:
        st.warning("⚠�? Please initialize the agent using the sidebar button before proceeding.")
        
        # Show example of what the system can do
        st.markdown("---")
        st.subheader("What This System Does")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### �? Application Review")
            st.write("Automatically reviews submitted permit applications for completeness and accuracy")
        
        with col2:
            st.markdown("#### ✅ Compliance Check")
            st.write("Verifies compliance with Denver food truck regulations and safety requirements")
        
        with col3:
            st.markdown("#### 📊 Scorecard")
            st.write("Generates detailed scorecards with actionable feedback and next steps")
        
        st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["�? Submit Application", "�?� Ask Questions", "📊 Evaluation History"])
    
    # ==================== TAB 1: Submit Application ====================
    with tab1:
        st.header("Submit Permit Application")
        
        with st.form("application_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Business Information")
                business_name = st.text_input("Business Name *", help="Legal business name")
                operator_name = st.text_input("Operator Name *", help="Name of primary operator")
                vehicle_type = st.selectbox(
                    "Vehicle Type *",
                    ["Mobile Truck", "Mobile Trailer", "Cart", "Other"]
                )
                menu_items = st.text_area(
                    "Menu Items *",
                    help="List menu items (one per line)",
                    height=100
                )
                
                st.subheader("Commissary Information")
                commissary_name = st.text_input("Commissary Name *")
                commissary_address = st.text_area("Commissary Address *", height=80)
            
            with col2:
                st.subheader("Water System")
                clean_water = st.number_input(
                    "Clean Water Tank (gallons) *",
                    min_value=0,
                    value=20,
                    help="Minimum 10 gallons or 3 gallons/hour"
                )
                wastewater = st.number_input(
                    "Wastewater Tank (gallons) *",
                    min_value=0,
                    value=25,
                    help="Must be 15% larger than clean water tank"
                )
                hand_sink_width = st.number_input("Hand Sink Width (inches) *", min_value=0, value=10)
                hand_sink_length = st.number_input("Hand Sink Length (inches) *", min_value=0, value=10)
                water_temp = st.number_input(
                    "Hot Water Temperature (°F) *",
                    min_value=0,
                    max_value=140,
                    value=110,
                    help="Required: 100-120°F"
                )
                
                st.subheader("Equipment")
                has_hood = st.checkbox("Type I Hood with Fire Suppression")
                has_refrigeration = st.checkbox("Commercial Refrigeration")
                has_ventilation = st.checkbox("Adequate Ventilation System")
                cooking_equipment = st.multiselect(
                    "Cooking Equipment",
                    ["Griddle", "Grill", "Deep Fryer", "Oven", "Steamer", "Other"]
                )
            
            st.subheader("Operating Information")
            col3, col4 = st.columns(2)
            
            with col3:
                locations = st.text_area(
                    "Proposed Operating Locations *",
                    help="List proposed locations (one per line)",
                    height=80
                )
            
            with col4:
                hours = st.text_input("Hours of Operation *", value="11:00 AM - 8:00 PM")
            
            st.subheader("Documentation")
            documents_checklist = st.multiselect(
                "Documents Attached *",
                [
                    "Vehicle Registration",
                    "Insurance Certificate",
                    "Commissary Affidavit",
                    "Mobile Unit Floor Plan",
                    "Equipment Specification Sheets",
                    "Water System Diagram",
                    "Waste Disposal Plan",
                    "Certified Food Manager Certificate"
                ]
            )
            
            # Submit button
            submitted = st.form_submit_button("�? Evaluate Application", type="primary")
        
        if submitted:
            # Validate required fields
            if not all([business_name, operator_name, commissary_name, commissary_address, menu_items, locations]):
                st.error("�?� Please fill in all required fields marked with *")
            else:
                # Build application dictionary
                application = {
                    "business_name": business_name,
                    "operator_name": operator_name,
                    "vehicle_type": vehicle_type,
                    "commissary": commissary_name,
                    "commissary_address": commissary_address,
                    "water_system": {
                        "clean_water_tank_size": f"{clean_water} gallons",
                        "wastewater_tank_size": f"{wastewater} gallons",
                        "hand_sink_dimensions": f"{hand_sink_width}x{hand_sink_length} inches",
                        "hot_water_temperature": f"{water_temp}°F"
                    },
                    "equipment": {
                        "type_i_hood": has_hood,
                        "commercial_refrigeration": has_refrigeration,
                        "ventilation_system": has_ventilation,
                        "cooking_equipment": cooking_equipment
                    },
                    "menu": [item.strip() for item in menu_items.split("\n") if item.strip()],
                    "proposed_locations": [loc.strip() for loc in locations.split("\n") if loc.strip()],
                    "hours_of_operation": hours,
                    "documents_attached": documents_checklist
                }
                
                # Evaluate application
                with st.spinner("🔄 Evaluating application... This may take a moment."):
                    try:
                        evaluation = st.session_state.agent.evaluate_application(application)
                        st.session_state.evaluation_history.append({
                            "application": application,
                            "evaluation": evaluation,
                            "timestamp": st.session_state.agent.session_id
                        })
                        
                        # Display results
                        st.markdown("---")
                        st.header("📊 Evaluation Results")
                        
                        # Overall metrics
                        overall_score = evaluation.get("overall_score", 0)
                        recommendation = evaluation.get("recommendation", "NEEDS_REVIEW")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Overall Score", f"{overall_score}/100")
                        
                        with col2:
                            st.metric("Recommendation", recommendation)
                        
                        with col3:
                            if recommendation == "APPROVED":
                                status_emoji = "🟢"
                                status_text = "Approved"
                            elif recommendation == "NEEDS_REVISION":
                                status_emoji = "🟡"
                                status_text = "Needs Revision"
                            else:
                                status_emoji = "🔴"
                                status_text = "Rejected"
                            st.metric("Status", f"{status_emoji} {status_text}")
                        
                        # Category scores
                        if "categories" in evaluation:
                            st.subheader("Category Breakdown")
                            
                            for category, details in evaluation["categories"].items():
                                category_name = category.replace("_", " ").title()
                                score = details.get("score", 0)
                                
                                with st.expander(f"{category_name}: {score}/100"):
                                    # Progress bar
                                    st.progress(score / 100)
                                    
                                    # Findings
                                    if details.get("findings"):
                                        st.markdown("**Findings:**")
                                        for finding in details["findings"]:
                                            st.markdown(f"- {finding}")
                                    
                                    # Required actions
                                    if details.get("required_actions"):
                                        st.markdown("**Required Actions:**")
                                        for action in details["required_actions"]:
                                            st.markdown(f"✓ {action}")
                        
                        # Summary
                        if "summary" in evaluation:
                            st.subheader("Summary")
                            st.info(evaluation["summary"])
                        
                        # Next steps
                        if "next_steps" in evaluation:
                            st.subheader("Next Steps")
                            for i, step in enumerate(evaluation["next_steps"], 1):
                                st.markdown(f"{i}. {step}")
                        
                        # Raw response (for debugging)
                        if "raw_response" in evaluation:
                            with st.expander("🔧 Debug: Raw Response"):
                                st.text(evaluation["raw_response"])
                        
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        st.exception(e)
    
    # ==================== TAB 2: Ask Questions ====================
    with tab2:
        st.header("Ask Questions About Permit Requirements")
        st.markdown("Ask any questions about Denver food truck permit regulations")
        
        # Question input
        question = st.text_area(
            "Your Question",
            placeholder="E.g., What are the water tank requirements for a food truck?",
            height=100
        )
        
        if st.button("Get Answer", type="primary"):
            if not question:
                st.warning("Please enter a question")
            else:
                with st.spinner("Searching regulations..."):
                    try:
                        answer = st.session_state.agent.query_with_rag(question)
                        
                        st.markdown("---")
                        st.subheader("Answer")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Common questions
        st.markdown("---")
        st.subheader("Common Questions")
        
        common_questions = [
            "What are the water tank requirements for a food truck?",
            "Where can I operate my food truck in Denver?",
            "What fire safety equipment is required?",
            "What documents do I need to submit for a permit?",
            "What are the commissary requirements?",
            "How much does a food truck permit cost?",
            "What are the hand washing sink requirements?",
            "Can I operate in public parks?"
        ]
        
        for q in common_questions:
            if st.button(q, key=f"common_{q}"):
                with st.spinner("Searching regulations..."):
                    try:
                        answer = st.session_state.agent.query_with_rag(q)
                        st.markdown("---")
                        st.subheader("Answer")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # ==================== TAB 3: Evaluation History ====================
    with tab3:
        st.header("Evaluation History")
        
        if st.session_state.evaluation_history:
            st.success(f"Total Evaluations: {len(st.session_state.evaluation_history)}")
            
            for i, record in enumerate(reversed(st.session_state.evaluation_history), 1):
                evaluation = record["evaluation"]
                application = record["application"]
                
                # Summary card
                with st.expander(
                    f"#{len(st.session_state.evaluation_history) - i + 1}: {application.get('business_name', 'N/A')} - "
                    f"{evaluation.get('recommendation', 'N/A')} ({evaluation.get('overall_score', 0)}/100)"
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Application Details:**")
                        st.json(application)
                    
                    with col2:
                        st.markdown("**Evaluation Results:**")
                        st.json(evaluation)
            
            # Clear history button
            if st.button("🗑�? Clear History", type="secondary"):
                st.session_state.evaluation_history = []
                st.success("History cleared")
                st.rerun()
        else:
            st.info("No evaluations yet. Submit an application in the 'Submit Application' tab to see results here.")

if __name__ == "__main__":
    main()
