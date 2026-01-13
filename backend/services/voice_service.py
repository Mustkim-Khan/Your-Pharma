"""
Voice Service
Models: gpt-4o-mini-transcribe (STT), gpt-4o-mini-tts (TTS)
Purpose: Handle speech-to-text and text-to-speech for live voice UX
"""
import base64
import io
from openai import OpenAI
import os

from utils.tracing_utils import traceable, wrap_openai, tracing_context


class VoiceService:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.stt_model = "gpt-4o-mini-transcribe"
        self.tts_model = "gpt-4o-mini-tts"
    
    @traceable(run_type="chain", name="VoiceService.transcribe")
    def transcribe(self, audio_base64: str) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_base64: Base64 encoded audio data
        
        Returns:
            Transcribed text
        """
        tracing_context.update_current_observation(
            metadata={
                "model": self.stt_model,
                "input_audio_length": len(audio_base64)
            }
        )
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.webm"
            
            # Call OpenAI transcription API
            response = self.client.audio.transcriptions.create(
                model=self.stt_model,
                file=audio_file,
                language="en"
            )
            
            transcript = response.text
            
            tracing_context.update_current_observation(
                metadata={"transcript": transcript}
            )
            
            return transcript
            
        except Exception as e:
            tracing_context.update_current_observation(
                metadata={"error": str(e)}
            )
            return ""
    
    @traceable(run_type="chain", name="VoiceService.synthesize")
    def synthesize(self, text: str, voice: str = "nova") -> str:
        """
        Synthesize text to speech
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        
        Returns:
            Base64 encoded audio data
        """
        tracing_context.update_current_observation(
            metadata={
                "model": self.tts_model,
                "text": text,
                "voice": voice
            }
        )
        
        try:
            # Limit text length for TTS
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # Get audio bytes and encode to base64
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            tracing_context.update_current_observation(
                metadata={"audio_length": len(audio_base64)}
            )
            
            return audio_base64
            
        except Exception as e:
            tracing_context.update_current_observation(
                metadata={"error": str(e)}
            )
            return ""
    
    def process_voice_input(self, audio_base64: str) -> tuple[str, str]:
        """
        Process voice input: transcribe and prepare for agent processing
        
        Args:
            audio_base64: Base64 encoded audio
        
        Returns:
            Tuple of (transcript, error_message)
        """
        transcript = self.transcribe(audio_base64)
        if not transcript:
            return "", "I couldn't understand the audio. Please try speaking again."
        return transcript, ""
    
    def generate_voice_response(self, text: str) -> str:
        """
        Generate voice response from text
        
        Args:
            text: Text response to convert
        
        Returns:
            Base64 encoded audio
        """
        # Clean text for TTS (remove markdown, emojis, etc.)
        clean_text = text.replace("**", "").replace("*", "")
        clean_text = clean_text.replace("✅", "").replace("❌", "")
        clean_text = clean_text.replace("⚠️", "").replace("📋", "")
        clean_text = clean_text.replace("💰", "").replace("📅", "")
        clean_text = clean_text.replace("📦", "").replace("🚚", "")
        clean_text = clean_text.replace("🔄", "").replace("⏳", "")
        
        return self.synthesize(clean_text)


# Singleton instance
voice_service = VoiceService()