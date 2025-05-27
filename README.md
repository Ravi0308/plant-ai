# Plant Expert AI Chatbot (React + Python Backend)

A professional, modern web-based chatbot application for plant identification and information. This application uses a React frontend with TypeScript and a Python Flask backend with Google's Gemini AI to provide accurate information about plants and gardening.

## Features

- 🌿 Modern, professional UI with 3D plant visualization using React and Three.js
- 💬 Text-based chat interface for plant questions
- 🎤 Voice input capability for hands-free interaction
- 📷 Image upload for plant identification
- 🧠 AI-powered responses using Google Gemini
- 📚 Knowledge base built from plant information PDF

## Technologies Used

- **Frontend**: React, TypeScript, Three.js, `@react-three/fiber`, `@react-three/drei`, Axios, CSS3
- **Backend**: Python, Flask, LangChain
- **AI/ML**: Google Gemini AI, FAISS vector database

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js (which includes npm) 14.x or higher
- Google API key for Gemini AI

### Installation

1.  **Clone the repository or download the files.**

2.  **Backend Setup (Python):**
    Navigate to the root project directory (`d:/ALPS/plant/`):
    ```bash
    pip install -r requirements.txt
    ```
    Make sure your `.env` file in the root directory contains your Google API key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

3.  **Frontend Setup (React):**
    Navigate to the `frontend` directory (`d:/ALPS/plant/frontend/`):
    ```bash
    npm install
    ```

### Running the Application

1.  **Start the Backend API Server:**
    In the root project directory (`d:/ALPS/plant/`), run:
    ```bash
    python api.py
    ```
    The backend will typically start on `http://localhost:5000`.

2.  **Start the Frontend Development Server:**
    In a **new terminal**, navigate to the `frontend` directory (`d:/ALPS/plant/frontend/`) and run:
    ```bash
    npm start
    ```
    The frontend will typically start on `http://localhost:3000` and open automatically in your browser.

## Usage

- Open `http://localhost:3000` in your browser.
- **Text Chat**: Type your plant-related questions in the input field.
- **Voice Input**: Click the microphone icon and speak your question.
- **Image Upload**: Click the image icon to upload a photo of a plant for identification.
- **3D Visualization**: Interact with the 3D plant model (rotate, zoom).

## File Structure

```
d:/ALPS/plant/
├── .env                  # Stores Google API Key
├── README.md             # This file
├── api.py                # Python Flask backend API
├── plant_info.pdf        # Source document for plant knowledge
├── requirements.txt      # Python dependencies
├── run.bat               # (Optional) Script to run backend
├── frontend/             # React frontend application
│   ├── public/
│   ├── src/
│   │   ├── components/   # React components (Chatbot, PlantModelViewer)
│   │   │   ├── Chatbot.tsx
│   │   │   ├── Chatbot.css
│   │   │   ├── PlantModelViewer.tsx
│   │   │   └── PlantModelViewer.css
│   │   ├── App.tsx       # Main App component
│   │   ├── App.css       # Main App styles
│   │   ├── index.tsx     # Entry point for React app
│   │   └── index.css     # Global styles
│   ├── package.json      # Frontend dependencies and scripts
│   └── tsconfig.json     # TypeScript configuration
└── chroma_index/         # FAISS vector store (auto-generated)
```

## Notes

- The application requires an internet connection to access Google's Gemini AI services.
- For optimal performance, use a modern browser (Chrome, Firefox, Edge).
- The 3D visualization requires WebGL support.

## License

This project is licensed under the MIT License.