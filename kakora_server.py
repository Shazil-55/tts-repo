#!/usr/bin/env python3
"""
Kokoro TTS API Server - Optimized for Single Voice (af_heart)
Lightweight server with minimal resource usage
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from kokoro import KPipeline
import soundfile as sf
import torch
import io
import tempfile
import uuid
import warnings
from datetime import datetime
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# Single pipeline for optimized performance
pipeline = None
VOICE = 'af_heart'

def initialize_pipeline():
    """Initialize single Kokoro pipeline for British accent"""
    global pipeline
    
    print("🔄 Initializing Kokoro pipeline...")
    
    # Force CPU usage
    torch.set_default_device('cpu')
    
    try:
        print(f"  🔄 Loading British English pipeline with {VOICE} voice...")
        pipeline = KPipeline(
            lang_code='b',  # British English only
            repo_id='hexgrad/Kokoro-82M'
        )
        print(f"  ✅ Pipeline ready with {VOICE} voice!")
        return True
    except Exception as e:
        print(f"  ❌ Failed to load pipeline: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if pipeline else 'error',
        'model': 'Kokoro-82M',
        'voice': VOICE,
        'accent': 'british',
        'device': 'CPU',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/voice', methods=['GET'])
def get_voice_info():
    """Get voice information"""
    return jsonify({
        'voice': VOICE,
        'name': 'Heart (Female)',
        'accent': 'british',
        'language': 'English (British)',
        'type': 'female'
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech with af_heart voice"""
    try:
        if not pipeline:
            return jsonify({'error': 'TTS pipeline not initialized'}), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        speed = data.get('speed', 1.0)
        
        # Validate parameters
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        if speed < 0.5 or speed > 2.0:
            return jsonify({'error': 'Speed must be between 0.5 and 2.0'}), 400
        
        print(f"🎤 Generating speech: voice={VOICE}, speed={speed}")
        
        # Generate audio
        generator = pipeline(text, voice=VOICE, speed=speed)
        
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
            download_name=f'kokoro_{VOICE}_{uuid.uuid4().hex[:8]}.wav'
        )
        
    except Exception as e:
        print(f"❌ Error in TTS: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Kokoro TTS API Server - Single Voice Edition")
    print(f"   Voice: {VOICE} (Heart - Female British)")
    
    if initialize_pipeline():
        print("\n📡 API Endpoints:")
        print("  GET  /health      - Health check")
        print("  GET  /voice       - Voice information")
        print("  POST /tts         - Generate speech (WAV file)")
        print("  POST /tts_base64  - Generate speech (base64)")
        print("  GET  /demo        - Web demo page")
        
        print("\n🌐 Starting server on http://localhost:5001")
        print("💡 Visit http://localhost:5001/demo for the web interface")
        print("⚡ Optimized for single voice - reduced memory usage!")
        
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("❌ Failed to initialize pipeline. Exiting.")
        exit(1)