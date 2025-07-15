#!/usr/bin/env python3
"""
Kokoro British Accent TTS API Server
Provides REST API endpoints for text-to-speech with British accent
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from kokoro import KPipeline
import soundfile as sf
import torch
import io
import os
import tempfile
import uuid
import warnings
from datetime import datetime
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# Global pipeline instances for different languages
pipelines = {}

def initialize_pipelines():
    """Initialize Kokoro pipelines for different accents"""
    print("🔄 Initializing Kokoro pipelines...")
    
    # Force CPU usage
    torch.set_default_device('cpu')
    
    # Initialize pipelines for different accents
    accent_configs = {
        'british': {'lang_code': 'b', 'name': 'British English'},
        'american': {'lang_code': 'a', 'name': 'American English'},
        'spanish': {'lang_code': 'e', 'name': 'Spanish'},
        'french': {'lang_code': 'f', 'name': 'French'},
        'italian': {'lang_code': 'i', 'name': 'Italian'},
    }
    
    for accent, config in accent_configs.items():
        try:
            print(f"  🔄 Loading {config['name']} pipeline...")
            pipelines[accent] = KPipeline(
                lang_code=config['lang_code'],
                repo_id='hexgrad/Kokoro-82M'  # Suppress repo_id warning
            )
            print(f"  ✅ {config['name']} pipeline ready!")
        except Exception as e:
            print(f"  ❌ Failed to load {config['name']}: {e}")
    
    print(f"✅ {len(pipelines)} pipelines initialized successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Kokoro-82M',
        'device': 'CPU',
        'available_accents': list(pipelines.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/accents', methods=['GET'])
def get_accents():
    """Get available accents"""
    accent_info = {
        'british': {'name': 'British English', 'lang_code': 'b'},
        'american': {'name': 'American English', 'lang_code': 'a'},
        'spanish': {'name': 'Spanish', 'lang_code': 'e'},
        'french': {'name': 'French', 'lang_code': 'f'},
        'italian': {'name': 'Italian', 'lang_code': 'i'},
    }
    
    available_accents = {}
    for accent in pipelines.keys():
        if accent in accent_info:
            available_accents[accent] = accent_info[accent]
    
    return jsonify({
        'accents': available_accents,
        'default': 'british'
    })

@app.route('/voices', methods=['GET'])
def get_voices():
    """Get available voices"""
    voices = {
        'female': ['af_heart', 'af_allison', 'af_monica', 'af_sara'],
        'male': ['am_aaron', 'am_eddie', 'am_harold', 'am_louis', 'am_mike'],
        'other': ['af_andrew']
    }
    return jsonify({
        'voices': voices,
        'default': 'af_allison',
        'note': 'All voices work with all accents'
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech with British accent (default)"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        accent = data.get('accent', 'british')  # Default to British
        voice = data.get('voice', 'af_heart')
        speed = data.get('speed', 1.0)
        
        # Validate parameters
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        if accent not in pipelines:
            return jsonify({'error': f'Accent "{accent}" not available. Available: {list(pipelines.keys())}'}), 400
        
        # Get the appropriate pipeline
        pipeline = pipelines[accent]
        
        # Generate audio
        generator = pipeline(text, voice=voice, speed=speed)
        
        # Collect all audio segments
        audio_segments = []
        for i, (gs, ps, audio) in enumerate(generator):
            audio_segments.append(audio)
        
        if not audio_segments:
            return jsonify({'error': 'No audio generated'}), 500
        
        # Concatenate audio segments
        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            final_audio = np.concatenate(audio_segments)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, final_audio, 24000)
        
        # Return audio file
        return send_file(
            temp_file.name,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'kokoro_{accent}_{uuid.uuid4().hex[:8]}.wav'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tts/british', methods=['POST'])
def british_tts():
    """Dedicated British accent TTS endpoint"""
    try:
        data = request.get_json()
        if not data:
            data = {}
        
        # Force British accent
        data['accent'] = 'british'
        
        # Use the main TTS function
        request.json = data
        return text_to_speech()
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tts_base64', methods=['POST'])
def text_to_speech_base64():
    """Convert text to speech and return base64 encoded audio"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        accent = data.get('accent', 'british')
        voice = data.get('voice', 'af_heart')
        speed = data.get('speed', 1.0)
        
        # Validate parameters
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if accent not in pipelines:
            return jsonify({'error': f'Accent "{accent}" not available'}), 400
        
        # Get the appropriate pipeline
        pipeline = pipelines[accent]
        
        # Generate audio
        generator = pipeline(text, voice=voice, speed=speed)
        
        # Collect all audio segments
        audio_segments = []
        for i, (gs, ps, audio) in enumerate(generator):
            audio_segments.append(audio)
        
        if not audio_segments:
            return jsonify({'error': 'No audio generated'}), 500
        
        # Concatenate audio segments
        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            final_audio = np.concatenate(audio_segments)
        
        # Convert to base64
        import base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, final_audio, 24000, format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            'audio_base64': audio_base64,
            'accent': accent,
            'voice': voice,
            'speed': speed,
            'sample_rate': 24000,
            'format': 'wav'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo', methods=['GET'])
def demo_page():
    """Simple demo page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kokoro British TTS Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .controls { margin: 10px 0; }
            select { padding: 5px; margin: 0 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🇬🇧 Kokoro British TTS Demo</h1>
            
            <textarea id="text" placeholder="Enter text to convert to speech...">Hello! This is a demonstration of the British accent text-to-speech system. The voice should sound naturally British.</textarea>
            
            <div class="controls">
                <label>Accent:</label>
                <select id="accent">
                    <option value="british">British English</option>
                    <option value="american">American English</option>
                    <option value="spanish">Spanish</option>
                    <option value="french">French</option>
                    <option value="italian">Italian</option>
                </select>
                
                <label>Voice:</label>
                <select id="voice">
                    <option value="af_heart">Female - Heart</option>
                    <option value="af_sara">Female - Sara</option>
                    <option value="af_allison">Female - Allison</option>
                    <option value="am_aaron">Male - Aaron</option>
                    <option value="am_mike">Male - Mike</option>
                </select>
                
                <label>Speed:</label>
                <select id="speed">
                    <option value="0.8">Slow (0.8x)</option>
                    <option value="1.0" selected>Normal (1.0x)</option>
                    <option value="1.2">Fast (1.2x)</option>
                </select>
            </div>
            
            <button onclick="generateSpeech()">🎤 Generate Speech</button>
            <button onclick="playAudio()">🔊 Play Audio</button>
            
            <div id="status"></div>
            <audio id="audio" controls style="width: 100%; margin-top: 10px;"></audio>
        </div>
        
        <script>
            let currentAudio = null;
            
            async function generateSpeech() {
                const text = document.getElementById('text').value;
                const accent = document.getElementById('accent').value;
                const voice = document.getElementById('voice').value;
                const speed = parseFloat(document.getElementById('speed').value);
                const status = document.getElementById('status');
                
                if (!text.trim()) {
                    status.innerHTML = '❌ Please enter some text';
                    return;
                }
                
                status.innerHTML = '🔄 Generating speech...';
                
                try {
                    const response = await fetch('/tts_base64', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, accent, voice, speed })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        const audioData = atob(result.audio_base64);
                        const audioBlob = new Blob([audioData], { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        const audio = document.getElementById('audio');
                        audio.src = audioUrl;
                        currentAudio = audioUrl;
                        
                        status.innerHTML = `✅ Speech generated successfully! (${accent} accent, ${voice} voice)`;
                    } else {
                        status.innerHTML = `❌ Error: ${result.error}`;
                    }
                } catch (error) {
                    status.innerHTML = `❌ Network error: ${error.message}`;
                }
            }
            
            function playAudio() {
                const audio = document.getElementById('audio');
                if (audio.src) {
                    audio.play();
                } else {
                    document.getElementById('status').innerHTML = '❌ No audio to play. Generate speech first.';
                }
            }
        </script>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    print("🚀 Starting Kokoro British TTS API Server...")
    initialize_pipelines()
    
    print("\n📡 API Endpoints:")
    print("  GET  /health       - Health check")
    print("  GET  /accents      - Available accents")
    print("  GET  /voices       - Available voices")
    print("  POST /tts          - Generate speech (default: British)")
    print("  POST /tts/british  - British accent TTS")
    print("  POST /tts_base64   - Generate speech (base64)")
    print("  GET  /demo         - Web demo page")
    
    print("\n🌐 Starting server on http://localhost:5000")
    print("💡 Visit http://localhost:5000/demo for a web interface")
    
    app.run(host='0.0.0.0', port=5001, debug=False)