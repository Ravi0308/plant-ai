# # """from flask import Flask, request, jsonify, send_from_directory
# # from flask_cors import CORS
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain_chroma import Chroma
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain.chains import ConversationalRetrievalChain, LLMChain
# # from langchain.memory import ConversationBufferWindowMemory
# # from dotenv import load_dotenv
# # import base64
# # import io
# # from PIL import Image
# # import traceback
# # import json

# # from pydantic import BaseModel, Field
# # from typing import List, Optional
# # from langchain.output_parsers import PydanticOutputParser

# # from google.generativeai.types import BlockedPromptException, HarmCategory, HarmBlockThreshold
# # from google.api_core import exceptions as google_exceptions

# # load_dotenv()

# # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# # if not GOOGLE_API_KEY:
# #     raise ValueError("GOOGLE_API_KEY environment variable is not set")
# # genai.configure(api_key=GOOGLE_API_KEY)

# # PDF_PATH = "plant_info.pdf"
# # CHROMA_DB_DIR = "chroma_index"

# # app = Flask(__name__, static_folder=".")

# # CORS(app, resources={
# #     r"/api/*": {
# #         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
# #         "methods": ["GET", "POST", "OPTIONS"],
# #         "allow_headers": ["Content-Type", "Authorization"]
# #     }
# # })

# # # --- Pydantic Models (Same as before) ---
# # class PlantIdentificationDetails(BaseModel):
# #     name: str = Field(description="Scientific name and common name of the plant")
# #     confidence: str = Field(description="Confidence level (High, Medium, Low)")
# #     description: Optional[str] = Field(None, description="Visual description")

# # class CareInstruction(BaseModel):
# #     category: str = Field(description="Care category (e.g., Watering, Light, Soil)")
# #     instruction: str = Field(description="Detailed instruction")

# # class PlantIdentificationOutput(BaseModel):
# #     identification: Optional[PlantIdentificationDetails] = Field(None)
# #     care_instructions: Optional[List[CareInstruction]] = Field(None)
# #     is_plant: bool = Field(True)
# #     message: Optional[str] = Field(None)

# #     def to_markdown(self, language_hint="English") -> str:
# #         not_a_plant_header = "## Not a Plant"
# #         identification_header = "## Plant Identification"
# #         name_label = "**Name:**"
# #         confidence_label = "**Confidence:**"
# #         description_label = "**Description:**"
# #         care_instructions_header = "## Care Instructions"
# #         note_header = "## Note"
# #         message_label = "**Message:**"

# #         if not self.is_plant:
# #             return f"{not_a_plant_header}\n*   {message_label} {self.message or 'This does not appear to be an image of a plant.'}"
# #         md_parts = []
# #         if self.identification:
# #             md_parts.append(identification_header)
# #             md_parts.append(f"*   {name_label} {self.identification.name}")
# #             md_parts.append(f"*   {confidence_label} {self.identification.confidence}")
# #             if self.identification.description:
# #                 md_parts.append(f"*   {description_label} {self.identification.description}")
# #         if self.care_instructions:
# #             md_parts.append(f"\n{care_instructions_header}")
# #             for care_item in self.care_instructions:
# #                 md_parts.append(f"*   **{care_item.category.strip().rstrip(':')}:** {care_item.instruction}")
# #         if self.message and not self.identification and not self.care_instructions:
# #             md_parts.append(f"\n{note_header}\n{self.message}")
# #         if not md_parts:
# #              return "## Plant Information\nNo specific details could be extracted, but the image appears to be a plant."
# #         return "\n\n".join(md_parts)

# # # --- Globals (Same as before) ---
# # vector_store = None
# # llm_safety_settings = {
# #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
# #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
# #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
# # }
# # chat_memory = ConversationBufferWindowMemory(
# #     k=10, # Increased k slightly for more context including image descriptions
# #     memory_key="chat_history", 
# #     return_messages=True, 
# #     output_key='answer' 
# # )

# # # --- Knowledge Base (Same as before) ---
# # def get_pdf_text(pdf_path):
# #     text = ""
# #     if not os.path.exists(pdf_path):
# #         print(f"Warning: PDF file not found at {pdf_path}. RAG features will be disabled.")
# #         return text
# #     try:
# #         with open(pdf_path, "rb") as file:
# #             pdf_reader = PdfReader(file)
# #             for page in pdf_reader.pages:
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     text += page_text
# #     except Exception as e:
# #         print(f"Error reading PDF {pdf_path}: {e}")
# #     return text

# # def get_text_chunks(text):
# #     if not text: return []
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks):
# #     if not text_chunks:
# #         print("No text chunks to create vector store from.")
# #         return None
# #     try:
# #         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
# #         if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
# #             print(f"Loading existing vector store from {CHROMA_DB_DIR}")
# #             return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
# #         else:
# #             print(f"Creating new vector store at {CHROMA_DB_DIR}")
# #             vs = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
# #             vs.persist() 
# #             return vs
# #     except Exception as e:
# #         print(f"Error creating/loading vector store: {e}")
# #         return None

# # def initialize_knowledge_base():
# #     global vector_store
# #     print("Attempting to initialize knowledge base...")
# #     if not os.path.exists(CHROMA_DB_DIR): os.makedirs(CHROMA_DB_DIR)
# #     raw_text = get_pdf_text(PDF_PATH)
# #     if not raw_text: return False
# #     text_chunks = get_text_chunks(raw_text)
# #     if not text_chunks: return False
# #     vector_store = get_vector_store(text_chunks)
# #     if vector_store is None: return False
# #     print("Knowledge base initialized/loaded successfully!")
# #     return True

# # # --- Conversational Chain with Memory (Same as before) ---
# # def get_conversational_rag_chain(memory_obj):
# #     global vector_store, llm_safety_settings
# #     retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None
# #     multilingual_instruction = """Your primary language for response is English. However, if the user's **current question** is clearly in Hindi, Gujarati, Hinglish (a mix of Hindi and English), or Ginglish (a mix of Gujarati and English), you MUST respond in that same language and style.
# # For example:
# # - User asks in Hindi: "यह पौधा कौन सा है?" -> You respond in Hindi.
# # - User asks in Gujarati: "આ છોડને કેટલું પાણી જોઈએ?" -> You respond in Gujarati.
# # - User asks in Hinglish: "Is plant ko sunlight pasand hai?" -> You respond in Hinglish.
# # If the user's question is in English or any other language, or if you are unsure of the user's language preference from the current question, respond in English.
# # """
# #     llm = ChatGoogleGenerativeAI(
# #         model="gemini-1.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY,
# #         safety_settings=llm_safety_settings, convert_system_message_to_human=True
# #     )
# #     if retriever:
# #         _template = f"""You are a plant expert assistant.
# # {multilingual_instruction}
# # Use the following pieces of retrieved context and the chat history to answer the question.
# # If you don't know the answer from the context or your general knowledge, just say that you don't know.
# # Do not try to make up an answer. If the question is not about plants, politely decline.
# # **Format your entire response using Markdown.**

# # Chat History (previous messages in this conversation):
# # {{chat_history}}

# # Context from knowledge base:
# # {{context}}

# # Current Question: {{question}}

# # Answer (in Markdown, following language instructions):"""
# #         chain = ConversationalRetrievalChain.from_llm(
# #             llm=llm, retriever=retriever, memory=memory_obj, return_source_documents=False,
# #             combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(_template)},
# #             verbose=False 
# #         )
# #     else: 
# #         print("Warning: Retriever not available. Using LLM with memory but no RAG context.")
# #         non_rag_template = f"""You are a plant expert assistant.
# # {multilingual_instruction}
# # If the question is not about plants, politely decline.
# # **Format your entire response using Markdown.**

# # Chat History (previous messages in this conversation):
# # {{chat_history}}
        
# # Current Question: {{question}}
# # Answer (in Markdown, following language instructions):"""
# #         prompt = ChatPromptTemplate.from_template(non_rag_template)
# #         chain = LLMChain(llm=llm, prompt=prompt, memory=memory_obj, verbose=False)
# #     return chain

# # # --- API Routes ---
# # @app.route('/')
# # def serve_index(): return "Plant Expert Backend is running."

# # @app.route('/api/health', methods=['GET'])
# # def health_check(): return jsonify({'status': 'healthy', 'message': 'Server is running'})

# # @app.route('/api/chat', methods=['POST'])
# # def chat():
# #     global chat_memory
# #     if not request.is_json: return jsonify({'error': 'Content-Type must be application/json'}), 400
# #     data = request.get_json()
# #     if not data: return jsonify({'error': 'No JSON data provided'}), 400
# #     user_message = data.get('message', '').strip()
# #     if not user_message: return jsonify({'error': 'No message provided'}), 400
# #     try:
# #         print(f"\n--- Chat Request --- User: {user_message}")
# #         # print(f"Memory BEFORE invoke: {chat_memory.load_memory_variables({})}")
# #         current_chain = get_conversational_rag_chain(chat_memory)
# #         if current_chain is None: return jsonify({'error': 'Chat service unavailable.'}), 500
        
# #         result = current_chain.invoke({"question": user_message})
# #         response_text = result.get('answer', result.get('text', "Sorry, I couldn't generate a response."))
        
# #         # print(f"Memory AFTER invoke: {chat_memory.load_memory_variables({})}")
# #         return jsonify({"sections": [{"title": "Response", "items": [response_text]}], "source": "Plant LLM"})
# #     except BlockedPromptException as e: return jsonify({'error': 'Message blocked. Please rephrase.'}), 400
# #     except google_exceptions.GoogleAPIError as e: return jsonify({'error': 'API service unavailable.'}), 503
# #     except Exception as e:
# #         print(f"Chat error: {e}"); traceback.print_exc()
# #         return jsonify({'error': 'Internal server error during chat.'}), 500

# # @app.route('/api/identify-plant', methods=['POST'])
# # def identify_plant():
# #     global llm_safety_settings, chat_memory # Added chat_memory here
# #     if not request.is_json: return jsonify({'error': 'Content-Type must be application/json'}), 400
# #     data = request.get_json()
# #     if not data: return jsonify({'error': 'No JSON data provided'}), 400

# #     image_data_b64 = data.get('image', '')
# #     user_text = data.get('text', '')
# #     frontend_language_hint = data.get('languageHint', 'english') # Default to English

# #     if not image_data_b64: return jsonify({'error': 'No image provided'}), 400

# #     try:
# #         image_bytes = base64.b64decode(image_data_b64)
# #         pil_image = Image.open(io.BytesIO(image_bytes))
# #     except Exception as e: return jsonify({'error': 'Invalid image data format.'}), 400

# #     language_map = {
# #         "hindi": "Hindi", "gujarati": "Gujarati",
# #         "hinglish": "Hinglish (a mix of Hindi and English)",
# #         "ginglish": "Ginglish (a mix of Gujarati and English)",
# #         "english": "English"
# #     }
# #     target_language = language_map.get(frontend_language_hint.lower(), "English")
# #     language_instruction = f"You MUST respond in {target_language}. All textual content MUST be in {target_language}."
# #     if target_language != "English":
# #         language_instruction += f" If absolutely unable for a specific part, English is a last resort for that part only."

# #     parser = PydanticOutputParser(pydantic_object=PlantIdentificationOutput)
# #     format_instructions = parser.get_format_instructions()
# #     model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=llm_safety_settings)
# #     user_context_text = f"User's question/context about the image: \"{user_text}\"\n\n" if user_text else ""

# #     prompt_text = f"{user_context_text}You are a plant ID expert. Analyze the image. {language_instruction}\n{format_instructions}\nResponse MUST be a single valid JSON object. No extra text."
    
# #     raw_llm_response_text = None
# #     markdown_response = "Error: Could not identify the plant or process the request." # Default error response
# #     status_code = 500

# #     try:
# #         print(f"Sending request to Gemini for plant ID (lang: {target_language}, expecting JSON)...")
# #         response_object = model.generate_content([pil_image, prompt_text]) 
# #         raw_llm_response_text = response_object.text
# #         print(f"Gemini Raw JSON Response for identify-plant:\n---\n{raw_llm_response_text}\n---")

# #         cleaned_text = raw_llm_response_text.strip()
# #         if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
# #         if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
# #         cleaned_text = cleaned_text.strip()
        
# #         parsed_output = parser.parse(cleaned_text)
# #         markdown_response = parsed_output.to_markdown(language_hint=target_language)

# #         # --- ADDING IMAGE IDENTIFICATION TO CHAT MEMORY ---
# #         synthetic_user_input_for_memory = f"User uploaded an image with the accompanying text: '{user_text if user_text else '[No text provided with image]'}'"
# #         # The 'output' is what the bot "said" about the image.
# #         chat_memory.save_context(
# #             {"input": synthetic_user_input_for_memory}, # What the "user" effectively did/said
# #             {"answer": markdown_response} # What the "bot" responded about the image
# #         )
# #         print(f"Saved image ID context to chat_memory. User: '{synthetic_user_input_for_memory[:100]}...', Bot: '{markdown_response[:100]}...'")
# #         # ----------------------------------------------------
        
# #         return jsonify({'response': markdown_response, 'status': 'success'})

# #     except Exception as e: # Broad catch for LLM errors, parsing errors, etc.
# #         print(f"Error in /identify-plant (parsing or other): {e}")
# #         traceback.print_exc()
        
# #         print(f"Falling back to simple Markdown for plant ID (lang: {target_language})...")
# #         fallback_language_instruction = f"You MUST respond in {target_language}. If unable, use English."
# #         fallback_prompt = f"{user_context_text}Analyze image. {fallback_language_instruction}\nIf plant: ID common name & key care (Water, Light, Soil).\nIf not plant: state clearly.\n**Format: simple Markdown.**"
# #         try:
# #             fallback_model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=llm_safety_settings)
# #             fallback_response_object = fallback_model.generate_content([pil_image, fallback_prompt])
# #             markdown_response = fallback_response_object.text

# #             # --- ADDING FALLBACK IMAGE IDENTIFICATION TO CHAT MEMORY ---
# #             synthetic_user_input_for_memory = f"User uploaded an image (fallback ID) with text: '{user_text if user_text else '[No text provided]'}'"
# #             chat_memory.save_context(
# #                 {"input": synthetic_user_input_for_memory},
# #                 {"answer": markdown_response}
# #             )
# #             print(f"Saved fallback image ID context to chat_memory. User: '{synthetic_user_input_for_memory[:100]}...', Bot: '{markdown_response[:100]}...'")
# #             # ---------------------------------------------------------

# #             return jsonify({'response': markdown_response, 'status': 'success_fallback_markdown'})
# #         except Exception as fallback_e:
# #             print(f"Fallback Markdown generation also failed: {fallback_e}")
# #             return jsonify({'error': 'Failed to identify plant, and fallback also failed.'}), 500


# # @app.route('/api/initialize', methods=['POST'])
# # def initialize_route():
# #     try:
# #         success = initialize_knowledge_base()
# #         return jsonify({'status': 'success' if success else 'failure', 
# #                         'message': 'Knowledge base initialized/loaded' if success else 'Failed to initialize knowledge base'}), \
# #                200 if success else 500
# #     except Exception as e:
# #         return jsonify({'error': 'Server error during knowledge base initialization'}), 500

# # @app.errorhandler(404)
# # def not_found(error): return jsonify({'error': 'Endpoint not found'}), 404
# # @app.errorhandler(500)
# # def internal_error(error):
# #     print(f"Unhandled Internal Server Error: {error}"); traceback.print_exc()
# #     return jsonify({'error': 'An unexpected internal server error occurred.'}), 500

# # if __name__ == '__main__':
# #     print("Starting Flask server...")
# #     if initialize_knowledge_base(): print("Knowledge base ready.")
# #     else: print("Warning: Knowledge base initialization failed. RAG may not work.")
# #     app.run(debug=True, host='0.0.0.0', port=5000)

# #     """

# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import ConversationalRetrievalChain, LLMChain
# from langchain.memory import ConversationBufferWindowMemory
# from dotenv import load_dotenv
# import base64
# import io
# from PIL import Image
# import traceback
# import json

# from pydantic import BaseModel, Field
# from typing import List, Optional, Any # Added Any here

# from langchain.output_parsers import PydanticOutputParser
# from google.generativeai.types import BlockedPromptException, HarmCategory, HarmBlockThreshold
# from google.api_core import exceptions as google_exceptions

# # --- Wikipedia Integration Imports ---
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools import WikipediaQueryRun
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.callbacks import CallbackManagerForRetrieverRun
# from langchain_core.documents import Document
# # --- End Wikipedia Integration Imports ---


# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable is not set")
# genai.configure(api_key=GOOGLE_API_KEY)

# PDF_PATH = "plant_info.pdf" # Make sure this PDF exists or RAG from PDF will be disabled
# CHROMA_DB_DIR = "chroma_index"

# app = Flask(__name__, static_folder=".") # Assuming your frontend is in a 'static' folder or built into '.'

# CORS(app, resources={
#     r"/api/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"], # Adjust for your frontend
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"]
#     }
# })

# # --- Pydantic Models ---
# class PlantIdentificationDetails(BaseModel):
#     name: str = Field(description="Scientific name and common name of the plant")
#     confidence: str = Field(description="Confidence level (High, Medium, Low)")
#     description: Optional[str] = Field(None, description="Visual description")

# class CareInstruction(BaseModel):
#     category: str = Field(description="Care category (e.g., Watering, Light, Soil)")
#     instruction: str = Field(description="Detailed instruction")

# class PlantIdentificationOutput(BaseModel):
#     identification: Optional[PlantIdentificationDetails] = Field(None)
#     care_instructions: Optional[List[CareInstruction]] = Field(None)
#     is_plant: bool = Field(True)
#     message: Optional[str] = Field(None)

#     def to_markdown(self, language_hint="English") -> str:
#         not_a_plant_header = "## Not a Plant"
#         identification_header = "## Plant Identification"
#         name_label = "**Name:**"
#         confidence_label = "**Confidence:**"
#         description_label = "**Description:**"
#         care_instructions_header = "## Care Instructions"
#         note_header = "## Note"
#         message_label = "**Message:**"

#         if not self.is_plant:
#             return f"{not_a_plant_header}\n*   {message_label} {self.message or 'This does not appear to be an image of a plant.'}"
#         md_parts = []
#         if self.identification:
#             md_parts.append(identification_header)
#             md_parts.append(f"*   {name_label} {self.identification.name}")
#             md_parts.append(f"*   {confidence_label} {self.identification.confidence}")
#             if self.identification.description:
#                 md_parts.append(f"*   {description_label} {self.identification.description}")
#         if self.care_instructions:
#             md_parts.append(f"\n{care_instructions_header}")
#             for care_item in self.care_instructions:
#                 md_parts.append(f"*   **{care_item.category.strip().rstrip(':')}:** {care_item.instruction}")
#         if self.message and not self.identification and not self.care_instructions: # Only show message if no other details
#             md_parts.append(f"\n{note_header}\n{self.message}")
#         if not md_parts:
#              return "## Plant Information\nNo specific details could be extracted, but the image appears to be a plant."
#         return "\n\n".join(md_parts)

# # --- Globals ---
# vector_store = None
# llm_safety_settings = {
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
# }
# chat_memory = ConversationBufferWindowMemory(
#     k=10,
#     memory_key="chat_history",
#     return_messages=True,
#     output_key='answer'
# )

# # --- Wikipedia Tool & Classifier LLM ---
# wikipedia_tool = WikipediaQueryRun(
#     api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2500, load_all_available_meta=False)
# )

# plant_query_classifier_llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash", # Using flash for speed in classification
#     temperature=0.0,
#     google_api_key=GOOGLE_API_KEY,
#     safety_settings=llm_safety_settings,
#     convert_system_message_to_human=True
# )

# # --- Knowledge Base (PDF) ---
# def get_pdf_text(pdf_path):
#     text = ""
#     if not os.path.exists(pdf_path):
#         print(f"Warning: PDF file not found at {pdf_path}. RAG from PDF will be disabled.")
#         return text
#     try:
#         with open(pdf_path, "rb") as file:
#             pdf_reader = PdfReader(file)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text
#     except Exception as e:
#         print(f"Error reading PDF {pdf_path}: {e}")
#     return text

# def get_text_chunks(text):
#     if not text: return []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     if not text_chunks:
#         print("No text chunks to create vector store from.")
#         return None
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#         if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
#             print(f"Loading existing vector store from {CHROMA_DB_DIR}")
#             return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
#         else:
#             print(f"Creating new vector store at {CHROMA_DB_DIR}")
#             vs = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
#             vs.persist()
#             return vs
#     except Exception as e:
#         print(f"Error creating/loading vector store: {e}")
#         return None

# def initialize_knowledge_base():
#     global vector_store
#     print("Attempting to initialize PDF knowledge base...")
#     if not os.path.exists(CHROMA_DB_DIR): os.makedirs(CHROMA_DB_DIR)
#     raw_text = get_pdf_text(PDF_PATH)
#     if not raw_text:
#         print("PDF raw text is empty. PDF RAG will be disabled.")
#         return False # Still return False, but not an error if PDF_PATH is just missing
#     text_chunks = get_text_chunks(raw_text)
#     if not text_chunks:
#         print("No text chunks from PDF. PDF RAG will be disabled.")
#         return False
#     vector_store = get_vector_store(text_chunks)
#     if vector_store is None:
#         print("Failed to create or load vector store from PDF.")
#         return False
#     print("PDF Knowledge base initialized/loaded successfully!")
#     return True

# # --- Custom Retrievers ---
# class WikipediaLangChainRetriever(BaseRetriever):
#     wikipedia_tool: WikipediaQueryRun
#     metadata_source: str = "Wikipedia"

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         try:
#             print(f"WikipediaRetriever: Querying Wikipedia for '{query}'")
#             wiki_text = self.wikipedia_tool.run(query)
#             if wiki_text and \
#                "No good Wikipedia Search Result was found" not in wiki_text and \
#                "Could not find page" not in wiki_text.lower() and \
#                "may refer to:" not in wiki_text.lower() and \
#                "similar pages" not in wiki_text.lower(): # Added checks for common 'not found' or disambiguation messages
#                 # Add a prefix to clearly indicate the source in the combined context
#                 return [Document(page_content=f"Context from {self.metadata_source}: {wiki_text}",
#                                  metadata={"source": self.metadata_source, "query": query})]
#             print(f"WikipediaRetriever: No relevant content found for '{query}' or content was a 'not found' message.")
#             return []
#         except Exception as e:
#             print(f"Error during Wikipedia retrieval for query '{query}': {e}")
#             return []

# wikipedia_retriever_instance = WikipediaLangChainRetriever(wikipedia_tool=wikipedia_tool)

# class CombinedRetriever(BaseRetriever):
#     pdf_retriever: Optional[BaseRetriever] # Can be None if PDF VS fails
#     wikipedia_retriever: WikipediaLangChainRetriever
#     plant_classifier_llm: ChatGoogleGenerativeAI

#     def _is_plant_related_query(self, query: str) -> bool:
#         # Simple keyword check first for speed, then LLM if keywords match
#         plant_keywords = ["plant", "tree", "flower", "garden", "botany", "leaf", "root", "soil", "water", "sunlight"]
#         if not any(keyword in query.lower() for keyword in plant_keywords):
#              print(f"Plant query classification (keyword pre-filter) for '{query}': NO (failed keyword check)")
#              return False # Faster exit for clearly non-plant queries

#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", "You are a query classifier. Determine if the user's query is primarily about plants, trees, flowers, gardening, or botany. Respond with only 'YES' or 'NO'."),
#             ("human", "{user_query}")
#         ])
#         chain = prompt_template | self.plant_classifier_llm
#         try:
#             response = chain.invoke({"user_query": query})
#             answer = response.content.strip().upper()
#             print(f"Plant query classification (LLM) for '{query}': {answer}")
#             return answer == "YES"
#         except Exception as e:
#             print(f"Error classifying query '{query}': {e}")
#             return False # Default to not plant-related on error

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         all_docs = []

#         # 1. Get documents from PDF store
#         if self.pdf_retriever:
#             print(f"CombinedRetriever: Fetching from PDF for: {query}")
#             try:
#                 pdf_docs = self.pdf_retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
#                 all_docs.extend(pdf_docs)
#                 print(f"CombinedRetriever: Found {len(pdf_docs)} docs from PDF.")
#             except Exception as e:
#                 print(f"CombinedRetriever: Error fetching from PDF: {e}")
#         else:
#             print("CombinedRetriever: PDF retriever not available.")

#         # 2. Conditionally get documents from Wikipedia
#         if self._is_plant_related_query(query):
#             print(f"CombinedRetriever: Query classified as plant-related. Fetching from Wikipedia for: {query}")
#             try:
#                 wiki_docs = self.wikipedia_retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
#                 all_docs.extend(wiki_docs)
#                 print(f"CombinedRetriever: Found {len(wiki_docs)} docs from Wikipedia.")
#             except Exception as e:
#                 print(f"CombinedRetriever: Error fetching from Wikipedia: {e}")
#         else:
#             print(f"CombinedRetriever: Query not classified as plant-related. Skipping Wikipedia for: {query}")

#         unique_docs = []
#         seen_contents_hashes = set() # Use hashes for potentially long contents
#         for doc in all_docs:
#             content_hash = hash(doc.page_content)
#             if content_hash not in seen_contents_hashes:
#                 unique_docs.append(doc)
#                 seen_contents_hashes.add(content_hash)
        
#         print(f"CombinedRetriever: Total unique documents returned: {len(unique_docs)}")
#         return unique_docs

# # --- Conversational Chain with Memory & Combined Retriever ---
# def get_conversational_rag_chain(memory_obj):
#     global vector_store, llm_safety_settings, plant_query_classifier_llm, wikipedia_retriever_instance

#     primary_pdf_retriever = None
#     if vector_store:
#         primary_pdf_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#     else:
#         print("Warning: PDF vector_store not available. PDF retrieval will be skipped.")

#     combined_retriever = CombinedRetriever(
#         pdf_retriever=primary_pdf_retriever,
#         wikipedia_retriever=wikipedia_retriever_instance,
#         plant_classifier_llm=plant_query_classifier_llm
#     )

#     multilingual_instruction = """Your primary language for response is English. However, if the user's **current question** is clearly in Hindi, Gujarati, Hinglish (a mix of Hindi and English), or Ginglish (a mix of Gujarati and English), you MUST respond in that same language and style.
# For example:
# - User asks in Hindi: "यह पौधा कौन सा है?" -> You respond in Hindi.
# - User asks in Gujarati: "આ છોડને કેટલું પાણી જોઈએ?" -> You respond in Gujarati.
# - User asks in Hinglish: "Is plant ko sunlight pasand hai?" -> You respond in Hinglish.
# If the user's question is in English or any other language, or if you are unsure of the user's language preference from the current question, respond in English.
# """
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash", # or gemini-1.5-pro for potentially better reasoning with combined context
#         temperature=0.6, # Slightly lower temp if context is good
#         google_api_key=GOOGLE_API_KEY,
#         safety_settings=llm_safety_settings,
#         convert_system_message_to_human=True
#     )

#     _template = f"""You are a plant expert assistant.
# {multilingual_instruction}
# Use the following pieces of retrieved context (which may come from a PDF knowledge base or Wikipedia) and the chat history to answer the question.
# Prioritize information from the PDF knowledge base if available and relevant. If not, or if Wikipedia provides more specific or better details for the user's query, use that.
# If the context contains information from both PDF and Wikipedia for the same plant, try to synthesize a comprehensive answer. If they conflict, state the conflict or lean on the more authoritative source if discernible (e.g., scientific description from Wikipedia vs. general care from PDF).
# If you don't know the answer from the provided context or your general knowledge, just say that you don't know.
# Do not try to make up an answer. If the question is not about plants, politely decline.
# **Format your entire response using Markdown.**

# Chat History (previous messages in this conversation):
# {{chat_history}}

# Context from knowledge base and/or Wikipedia:
# {{context}}

# Current Question: {{question}}

# Answer (in Markdown, following language instructions):"""
    
#     # Check if any retriever source is actually configured (even if pdf_retriever itself is None, combined_retriever exists)
#     # The combined_retriever will function (empty if no sources)
#     # The key is that ConversationalRetrievalChain expects a retriever.
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=combined_retriever,
#         memory=memory_obj,
#         return_source_documents=False, # Set to True if you want to debug sources
#         combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(_template)},
#         verbose=False # Set to True for more detailed logging from LangChain
#     )
#     return chain


# # --- API Routes ---
# @app.route('/')
# def serve_index():
#     # If you have a built frontend in a 'static' folder (e.g., from npm run build)
#     # and app = Flask(__name__, static_folder="static", static_url_path="")
#     # you might serve index.html here. Otherwise, a simple message is fine.
#     return "Plant Expert Backend is running. PDF RAG available: " + str(vector_store is not None)

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({
#         'status': 'healthy',
#         'message': 'Server is running',
#         'pdf_rag_initialized': vector_store is not None
#     })

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     global chat_memory
#     if not request.is_json:
#         return jsonify({'error': 'Content-Type must be application/json'}), 400
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'No JSON data provided'}), 400

#     user_message = data.get('message', '').strip()
#     if not user_message:
#         return jsonify({'error': 'No message provided'}), 400

#     try:
#         print(f"\n--- Chat Request --- User: {user_message}")
#         current_chain = get_conversational_rag_chain(chat_memory)
#         if current_chain is None: # Should not happen with the new structure but good check
#             return jsonify({'error': 'Chat service unavailable (chain not initialized).'}), 500
        
#         # print(f"Memory BEFORE invoke: {chat_memory.load_memory_variables({})}")
#         result = current_chain.invoke({"question": user_message})
#         response_text = result.get('answer', result.get('text', "Sorry, I couldn't generate a response."))
#         # print(f"Memory AFTER invoke: {chat_memory.load_memory_variables({})}")
        
#         return jsonify({"sections": [{"title": "Response", "items": [response_text]}], "source": "Plant LLM"})

#     except BlockedPromptException as e:
#         print(f"Chat error - BlockedPromptException: {e}")
#         return jsonify({'error': f'Your message was blocked due to safety concerns. Please rephrase. Details: {e}'}), 400
#     except google_exceptions.GoogleAPIError as e:
#         print(f"Chat error - GoogleAPIError: {e}")
#         return jsonify({'error': f'Google API service unavailable or encountered an error: {e.message}'}), 503
#     except Exception as e:
#         print(f"Chat error: {e}")
#         traceback.print_exc()
#         return jsonify({'error': 'Internal server error during chat.'}), 500

# @app.route('/api/identify-plant', methods=['POST'])
# def identify_plant():
#     global llm_safety_settings, chat_memory
#     if not request.is_json:
#         return jsonify({'error': 'Content-Type must be application/json'}), 400
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'No JSON data provided'}), 400

#     image_data_b64 = data.get('image', '')
#     user_text = data.get('text', '') # Accompanying text from user
#     frontend_language_hint = data.get('languageHint', 'english')

#     if not image_data_b64:
#         return jsonify({'error': 'No image provided'}), 400

#     try:
#         image_bytes = base64.b64decode(image_data_b64)
#         pil_image = Image.open(io.BytesIO(image_bytes))
#     except Exception as e:
#         print(f"Error decoding image: {e}")
#         return jsonify({'error': 'Invalid image data format.'}), 400

#     language_map = {
#         "hindi": "Hindi", "gujarati": "Gujarati",
#         "hinglish": "Hinglish (a mix of Hindi and English)",
#         "ginglish": "Ginglish (a mix of Gujarati and English)",
#         "english": "English"
#     }
#     target_language = language_map.get(frontend_language_hint.lower(), "English")
#     language_instruction = f"You MUST respond in {target_language}. All textual content (names, descriptions, care instructions, messages) MUST be in {target_language}."
#     if target_language != "English":
#         language_instruction += f" If absolutely unable for a specific part, English is a last resort for that part only, but strive for complete {target_language}."

#     parser = PydanticOutputParser(pydantic_object=PlantIdentificationOutput)
#     format_instructions = parser.get_format_instructions()
#     model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=llm_safety_settings) # or 'gemini-pro-vision'
    
#     user_context_text = f"User's question or context about the image: \"{user_text}\"\n\n" if user_text else ""
#     prompt_text = (
#         f"{user_context_text}"
#         f"You are a plant identification expert. Analyze the provided image. "
#         f"{language_instruction}\n"
#         f"Follow these JSON format instructions precisely:\n{format_instructions}\n"
#         f"Your entire response MUST be a single, valid JSON object matching the Pydantic schema. No extra text or markdown formatting outside the JSON structure."
#         f"If the image is not a plant, set 'is_plant' to false and provide a message in the '{target_language}' language explaining this. "
#         f"If it is a plant but you cannot identify it, set 'is_plant' to true, and provide a message in the 'message' field in {target_language}."
#     )
    
#     raw_llm_response_text = None # For debugging
#     try:
#         print(f"Sending request to Gemini for plant ID (lang: {target_language}, expecting JSON)...")
#         # Ensure the model can handle images and text. 'gemini-1.5-flash' and 'gemini-pro-vision' can.
#         response_object = model.generate_content([prompt_text, pil_image]) # Order might matter for some models
#         raw_llm_response_text = response_object.text
#         print(f"Gemini Raw Response for identify-plant:\n---\n{raw_llm_response_text}\n---")

#         # Clean the response (remove potential markdown code block fences)
#         cleaned_text = raw_llm_response_text.strip()
#         if cleaned_text.startswith("```json"):
#             cleaned_text = cleaned_text[7:]
#         if cleaned_text.endswith("```"):
#             cleaned_text = cleaned_text[:-3]
#         cleaned_text = cleaned_text.strip()
        
#         parsed_output = parser.parse(cleaned_text)
#         markdown_response = parsed_output.to_markdown(language_hint=target_language)

#         # Add to chat memory
#         synthetic_user_input_for_memory = f"User uploaded an image for identification. User's accompanying text: '{user_text if user_text else '[No text provided]'}'"
#         memory_bot_response = f"Identified image. Result: {parsed_output.identification.name if parsed_output.identification else 'Not identified or not a plant.'}"
#         if not parsed_output.is_plant:
#             memory_bot_response = f"Image analysis: Not a plant. Message: {parsed_output.message}"
#         elif parsed_output.message and not parsed_output.identification: # Plant, but not identified
#              memory_bot_response = f"Image analysis: Could not identify plant. Message: {parsed_output.message}"


#         chat_memory.save_context(
#             {"input": synthetic_user_input_for_memory},
#             {"answer": memory_bot_response} # Keep memory response concise
#         )
#         print(f"Saved image ID context to chat_memory. User: '{synthetic_user_input_for_memory[:100]}...', Bot: '{memory_bot_response[:100]}...'")
        
#         return jsonify({'response': markdown_response, 'status': 'success'})

#     except (BlockedPromptException, google_exceptions.GoogleAPIError) as direct_api_error:
#         print(f"API Error in /identify-plant: {direct_api_error}")
#         return jsonify({'error': f'API error during plant identification: {direct_api_error}'}), 503
#     except Exception as e:
#         print(f"Error in /identify-plant (could be parsing, LLM error, etc.): {e}")
#         if raw_llm_response_text:
#              print(f"Failed to parse this raw response for Pydantic: {raw_llm_response_text}")
#         traceback.print_exc()
        
#         # Fallback to simple markdown generation if Pydantic parsing fails
#         print(f"Falling back to simple Markdown for plant ID (lang: {target_language})...")
#         fallback_language_instruction = f"You MUST respond in {target_language}. If unable, use English as a last resort only for parts you cannot translate."
#         fallback_prompt = (
#             f"{user_context_text}"
#             f"Analyze the provided image. {fallback_language_instruction}\n"
#             f"If it is a plant, identify its common name and provide key care instructions (e.g., Watering, Light, Soil).\n"
#             f"If it is not a plant, clearly state that.\n"
#             f"**Format your entire response using simple Markdown.** No JSON."
#         )
#         try:
#             # Use a model that can handle images and text for fallback
#             fallback_model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=llm_safety_settings)
#             fallback_response_object = fallback_model.generate_content([fallback_prompt, pil_image]) # Order
#             markdown_response = fallback_response_object.text

#             synthetic_user_input_for_memory = f"User uploaded an image (fallback ID). User text: '{user_text if user_text else '[No text provided]'}'"
#             chat_memory.save_context(
#                 {"input": synthetic_user_input_for_memory},
#                 {"answer": f"Image analysis (fallback): {markdown_response[:150]}..."} # Keep memory concise
#             )
#             print(f"Saved fallback image ID context to chat_memory. User: '{synthetic_user_input_for_memory[:100]}...', Bot: '{markdown_response[:100]}...'")
#             return jsonify({'response': markdown_response, 'status': 'success_fallback_markdown'})
#         except Exception as fallback_e:
#             print(f"Fallback Markdown generation also failed: {fallback_e}")
#             return jsonify({'error': 'Failed to identify plant, and fallback also failed.'}), 500


# @app.route('/api/initialize', methods=['POST'])
# def initialize_route():
#     try:
#         success = initialize_knowledge_base()
#         return jsonify({'status': 'success' if success else 'failure',
#                         'message': 'PDF Knowledge base initialized/loaded' if success else 'Failed to initialize PDF knowledge base. PDF RAG may be disabled.'}), \
#                200 # Always 200, status in JSON indicates success/failure of init
#     except Exception as e:
#         print(f"Error in /api/initialize: {e}")
#         traceback.print_exc()
#         return jsonify({'error': 'Server error during knowledge base initialization'}), 500

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': 'Endpoint not found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     # Log the error for server-side debugging
#     print(f"Unhandled Internal Server Error: {error}")
#     traceback.print_exc()
#     return jsonify({'error': 'An unexpected internal server error occurred.'}), 500

# if __name__ == '__main__':
#     print("Starting Flask server...")
#     # Initialize PDF knowledge base at startup
#     pdf_initialized = initialize_knowledge_base()
#     if pdf_initialized:
#         print("PDF Knowledge base ready.")
#     else:
#         print("Warning: PDF Knowledge base initialization failed or PDF not found. RAG from PDF will be disabled.")
#     print("Wikipedia retriever is configured.")
#     print(f"Serving on http://0.0.0.0:5000")
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import base64
import io
from PIL import Image
import traceback
import json

from pydantic import BaseModel, Field
from typing import List, Optional, Any

from langchain.output_parsers import PydanticOutputParser
from google.generativeai.types import BlockedPromptException, HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__, static_folder=".")

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# --- Pydantic Models (for Image Identification) ---
class PlantIdentificationDetails(BaseModel):
    name: str = Field(description="Scientific name and common name of the plant")
    confidence: str = Field(description="Confidence level (High, Medium, Low)")
    description: Optional[str] = Field(None, description="Visual description")

class CareInstruction(BaseModel):
    category: str = Field(description="Care category (e.g., Watering, Light, Soil)")
    instruction: str = Field(description="Detailed instruction")

class PlantIdentificationOutput(BaseModel):
    identification: Optional[PlantIdentificationDetails] = Field(None)
    care_instructions: Optional[List[CareInstruction]] = Field(None)
    is_plant: bool = Field(True)
    message: Optional[str] = Field(None)

    def to_markdown(self, language_hint="English") -> str:
        not_a_plant_header = "## Not a Plant"
        identification_header = "## Plant Identification"
        name_label = "**Name:**"
        confidence_label = "**Confidence:**"
        description_label = "**Description:**"
        care_instructions_header = "## Care Instructions"
        note_header = "## Note"
        message_label = "**Message:**"

        if not self.is_plant:
            return f"{not_a_plant_header}\n*   {message_label} {self.message or 'This does not appear to be an image of a plant.'}"
        md_parts = []
        if self.identification:
            md_parts.append(identification_header)
            md_parts.append(f"*   {name_label} {self.identification.name}")
            md_parts.append(f"*   {confidence_label} {self.identification.confidence}")
            if self.identification.description:
                md_parts.append(f"*   {description_label} {self.identification.description}")
        if self.care_instructions:
            md_parts.append(f"\n{care_instructions_header}")
            for care_item in self.care_instructions:
                md_parts.append(f"*   **{care_item.category.strip().rstrip(':')}:** {care_item.instruction}")
        if self.message and not self.identification and not self.care_instructions:
            md_parts.append(f"\n{note_header}\n{self.message}")
        if not md_parts:
             return "## Plant Information\nNo specific details could be extracted, but the image appears to be a plant."
        return "\n\n".join(md_parts)

# --- Globals ---
llm_safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

chat_memory = ConversationBufferWindowMemory(
    k=10,
    memory_key="chat_history",
    input_key="question",
    return_messages=True,
    output_key='answer'
)

# --- Plant Query Classifier LLM ---
plant_query_classifier_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY,
    safety_settings=llm_safety_settings,
)

def is_plant_related_query(query: str, previous_bot_response: Optional[str] = None) -> bool:
    """
    Classifies if a query is plant-related.
    It now has a stronger prompt for the LLM part to understand follow-ups.
    """
    plant_keywords = [
        "plant", "tree", "flower", "garden", "gardening", "botany", "botanical",
        "leaf", "root", "stem", "seed", "fruit", "soil", "water", "sunlight", "fertilizer",
        "herb", "shrub", "cactus", "succulent", "bloom", "care", "grow", "cultivate"
    ]
    query_lower = query.lower()

    # If it has explicit plant keywords, it's likely plant-related.
    if any(keyword in query_lower for keyword in plant_keywords):
        print(f"Plant query classification (keyword pre-filter) for '{query}': YES (explicit keyword)")
        return True

    # For other cases (like "summarize this"), let the LLM decide based on context.
    # The LLM prompt for the classifier is now more robust.
    system_message_for_classifier = """You are a query classifier for a Plant AI assistant.
Your task is to determine if the user's current query is:
1.  Primarily about plants, trees, flowers, gardening, botany, or plant care.
2.  A direct follow-up question (e.g., asking for a summary, elaboration, or clarification) to the AI's immediately preceding plant-related response.

Consider the user's query. If it's a generic follow-up like "summarize that" or "tell me more", it's only "YES" if the context implies it's about a previous plant discussion.
If the query introduces a new topic that is clearly not about plants (e.g., "what's the weather?", "tell me a joke about cats"), then it's "NO".

Respond with only 'YES' or 'NO'."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_for_classifier),
        ("human", "User's current query: \"{user_query}\"")
    ])
    chain = prompt_template | plant_query_classifier_llm
    try:
        # We don't pass previous_bot_response directly to this classifier yet,
        # but the main chat chain will have the full history.
        response = chain.invoke({"user_query": query})
        answer = response.content.strip().upper()
        print(f"Plant query classification (LLM) for '{query}': {answer}")
        return answer == "YES"
    except Exception as e:
        print(f"Error classifying query '{query}': {e}")
        traceback.print_exc()
        return True # Default to True on error to let the main chain try

# --- Conversational Chain ---
def get_conversational_chain(memory_obj):
    multilingual_instruction = """Your primary language for response is English. However, if the user's **current question** is clearly in Hindi, Gujarati, Hinglish (a mix of Hindi and English), or Ginglish (a mix of Gujarati and English), you MUST respond in that same language and style.
For example:
- User asks in Hindi: "यह पौधा कौन सा है?" -> You respond in Hindi.
If the user's question is in English or any other language, or if you are unsure of the user's language preference from the current question, respond in English.
"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
        safety_settings=llm_safety_settings,
    )

    _template = f"""You are a helpful and knowledgeable Plant AI assistant.
{multilingual_instruction}
Your primary role is to discuss plants, gardening, botany, and related topics.
Use the chat history to understand the context of the conversation, especially for follow-up questions like "summarize that" or "tell me more".
If the user asks a question that is clearly not about plants and is not a follow-up to a plant discussion, politely state that you are a specialized Plant AI and can only discuss plant-related topics.
For example, if the chat history is about roses and the user asks "summarize that", you should summarize the rose discussion.
If the chat history is empty or about non-plant topics, and the user asks "what's the capital of France?", you should state your specialization.

**Format your entire response using Markdown.**

Chat History (previous messages in this conversation):
{{chat_history}}

Current Question: {{question}}

Answer (in Markdown, following language instructions):"""

    prompt = ChatPromptTemplate.from_template(_template)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory_obj,
        verbose=True,
        output_key='answer'
    )
    return chain

# --- API Routes ---
@app.route('/')
def serve_index():
    return "🌱 Plant Expert AI Backend (Simplified - Memory Fix) is running!"

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Simplified Plant AI (Memory Fix) is ready!'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_memory
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        print(f"\n--- Chat Request --- User: {user_message}")

        # The main conversational chain now ALWAYS runs.
        # Its prompt is designed to handle non-plant queries appropriately by stating its specialization.
        # The `is_plant_related_query` function can be seen as a potential pre-filter
        # for more complex scenarios in the future, but for now, let the main chain's prompt guide behavior.

        conversational_chain = get_conversational_chain(chat_memory)
        # The chain will use memory and also save the current interaction.
        # Its prompt now instructs it how to behave if the query (even with history) isn't plant-related.
        result = conversational_chain.invoke({"question": user_message})
        response_text = result.get('answer', "Sorry, I couldn't generate a response.")

        # For debugging memory:
        current_memory_vars = chat_memory.load_memory_variables({})
        # print(f"Current memory state after chat: {current_memory_vars}") # Uncomment to see memory

        return jsonify({
            "sections": [{"title": "🌱 Plant AI Response", "items": [response_text]}],
            "source": "Plant AI General Knowledge with Memory"
        })

    except BlockedPromptException as e:
        print(f"Chat error - BlockedPromptException: {e}")
        chat_memory.save_context({chat_memory.input_key: user_message}, {chat_memory.output_key: "My apologies, but your message was blocked due to safety concerns. Could you please rephrase?"})
        return jsonify({'error': f'Your message was blocked due to safety concerns. Please rephrase. Details: {e}'}), 400
    except google_exceptions.GoogleAPIError as e:
        print(f"Chat error - GoogleAPIError: {e}")
        chat_memory.save_context({chat_memory.input_key: user_message}, {chat_memory.output_key: "I'm having trouble connecting to my knowledge source right now. Please try again in a moment."})
        return jsonify({'error': f'Google API service unavailable or encountered an error: {e.message}'}), 503
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        chat_memory.save_context({chat_memory.input_key: user_message}, {chat_memory.output_key: "I encountered an internal error. Please try asking something else."})
        return jsonify({'error': 'Internal server error during chat.'}), 500


@app.route('/api/identify-plant', methods=['POST'])
def identify_plant():
    global llm_safety_settings, chat_memory
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    image_data_b64 = data.get('image', '')
    user_text = data.get('text', '')
    frontend_language_hint = data.get('languageHint', 'english')

    if not image_data_b64:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_bytes = base64.b64decode(image_data_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Error decoding image: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Invalid image data format.'}), 400

    language_map = {
        "hindi": "Hindi", "gujarati": "Gujarati",
        "hinglish": "Hinglish (a mix of Hindi and English)",
        "ginglish": "Ginglish (a mix of Gujarati and English)",
        "english": "English"
    }
    target_language = language_map.get(frontend_language_hint.lower(), "English")
    language_instruction = f"You MUST respond in {target_language}. All textual content MUST be in {target_language}."

    parser = PydanticOutputParser(pydantic_object=PlantIdentificationOutput)
    format_instructions = parser.get_format_instructions()
    model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=llm_safety_settings)

    user_context_text = f"User's question or context about the image: \"{user_text}\"\n\n" if user_text else ""
    prompt_text = (
        f"{user_context_text}"
        f"You are a plant identification expert. Analyze the provided image. "
        f"{language_instruction}\n"
        f"Follow these JSON format instructions precisely:\n{format_instructions}\n"
        f"Your entire response MUST be a single, valid JSON object. No extra text outside the JSON structure."
    )
    raw_llm_response_text = "Not available"

    try:
        print(f"Processing plant identification (lang: {target_language})")
        response_object = model.generate_content([prompt_text, pil_image])
        raw_llm_response_text = response_object.text
        print(f"Gemini Raw Response for identify-plant:\n---\n{raw_llm_response_text}\n---")

        cleaned_text = raw_llm_response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[len("```json"):]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")]
        cleaned_text = cleaned_text.strip()

        try:
            parsed_output = parser.parse(cleaned_text)
            markdown_response = parsed_output.to_markdown(language_hint=target_language)
            plant_name = parsed_output.identification.name if parsed_output.identification else "Unknown plant"
            status = 'success'
        except Exception as parse_error:
            print(f"Could not parse Pydantic output for plant ID: {parse_error}")
            markdown_response = f"## Plant Information ({target_language})\nWhile I couldn't structure all details perfectly, here's what I found:\n\n{cleaned_text}"
            plant_name = "Possibly identified plant (details might be in text)"
            status = 'success_partial_parse'

        synthetic_user_input = f"User uploaded an image for plant identification. {user_text if user_text else 'No additional context provided.'}"
        memory_response = f"Plant Identification Result: {plant_name}. Provided identification and care instructions based on the uploaded image."
        chat_memory.save_context(
            {chat_memory.input_key: synthetic_user_input},
            {chat_memory.output_key: memory_response}
        )
        return jsonify({'response': markdown_response, 'status': status})

    except Exception as e:
        print(f"Error in plant identification: {e}")
        print(f"Raw LLM Response (if available) was: {raw_llm_response_text}")
        traceback.print_exc()
        fallback_prompt = (
            f"{user_context_text}"
            f"Analyze this image. If it appears to be a plant, please identify it and provide "
            f"basic care instructions. Respond in {target_language}. Format your response clearly using Markdown."
        )
        try:
            fallback_model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=llm_safety_settings)
            fallback_response = fallback_model.generate_content([fallback_prompt, pil_image])
            markdown_response = fallback_response.text
            chat_memory.save_context(
                {chat_memory.input_key: f"User uploaded an image (fallback identification). {user_text}"},
                {chat_memory.output_key: "Provided plant identification using a fallback method."}
            )
            return jsonify({'response': markdown_response, 'status': 'success_fallback'})
        except Exception as fallback_e:
            print(f"Fallback identification also failed: {fallback_e}")
            traceback.print_exc()
            chat_memory.save_context(
                {chat_memory.input_key: f"User uploaded an image. {user_text}"},
                {chat_memory.output_key: "I'm sorry, I encountered an error trying to identify the plant from the image."}
            )
            return jsonify({'error': 'Failed to identify plant even with fallback.'}), 500

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"Unhandled Internal Server Error: {error}")
    traceback.print_exc()
    return jsonify({'error': 'An unexpected internal server error occurred.'}), 500

if __name__ == '__main__':
    print("🌱 Starting Simplified Plant AI Backend (Memory Fix)...")
    print("✅ Memory system active")
    print("✅ Plant query classification ready (though main chain handles topic gating)")
    print(f"🚀 Plant AI serving on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)