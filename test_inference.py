"""
Test script for RVC Inference Server.
Uploads a test audio file and performs voice conversion.
"""

import argparse
import io
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Check if requests is available
try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a simple test audio (sine wave) as WAV bytes.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
    
    Returns:
        WAV file as bytes
    """
    # Generate sine wave with varying frequency (simulates speech-like audio)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a more complex waveform
    freq1 = 220  # Base frequency
    freq2 = 330
    
    # Frequency modulation for variety
    mod = np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
    freq_mod = freq1 + 50 * mod
    
    # Generate audio
    audio = 0.3 * np.sin(2 * np.pi * freq_mod * t)
    audio += 0.2 * np.sin(2 * np.pi * freq2 * t)
    
    # Add envelope
    envelope = np.ones_like(t)
    attack = int(0.1 * sample_rate)
    release = int(0.2 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    audio *= envelope
    
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    
    # Create WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    
    buffer.seek(0)
    return buffer.read()


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    print("\n[TEST] Health Check")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Loaded models: {data['loaded_models']}")
        print(f"  Available models: {data['available_models']}")
        print(f"  Models dir: {data['models_dir']}")
        print("  [PASS] Health check successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        return False


def test_list_voices(base_url: str) -> list:
    """Test the voices endpoint."""
    print("\n[TEST] List Voices")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/voices", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"  Total voices: {data['total']}")
        for voice in data['voices']:
            loaded = "✓" if voice['loaded'] else "○"
            index = "+index" if voice['has_index'] else ""
            print(f"    [{loaded}] {voice['name']} {index}")
        
        print("  [PASS] Voices listed successfully")
        return [v['name'] for v in data['voices']]
    except Exception as e:
        print(f"  [FAIL] List voices failed: {e}")
        return []


def test_convert(
    base_url: str,
    voice_name: str,
    audio_file: str = None,
    f0_up_key: int = 0,
    f0_method: str = "harvest",
    output_file: str = None
) -> bool:
    """Test the convert endpoint."""
    print(f"\n[TEST] Convert Audio")
    print("-" * 40)
    print(f"  Voice: {voice_name}")
    print(f"  Pitch: {f0_up_key} semitones")
    print(f"  F0 Method: {f0_method}")
    
    try:
        # Prepare audio
        if audio_file and Path(audio_file).exists():
            print(f"  Input: {audio_file}")
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            filename = Path(audio_file).name
        else:
            print("  Input: Generated test audio (3s sine wave)")
            audio_bytes = generate_test_audio()
            filename = "test_audio.wav"
        
        # Send request
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/convert",
            files={"audio": (filename, audio_bytes, "audio/wav")},
            data={
                "voice_name": voice_name,
                "f0_up_key": f0_up_key,
                "f0_method": f0_method
            },
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"  [FAIL] Server returned {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        # Parse response headers
        processing_time = float(response.headers.get('X-Processing-Time', 0))
        input_duration = float(response.headers.get('X-Input-Duration', 0))
        output_sr = response.headers.get('X-Output-Sample-Rate', 'unknown')
        
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Input duration: {input_duration:.3f}s")
        print(f"  Output sample rate: {output_sr}")
        print(f"  Total request time: {elapsed:.3f}s")
        print(f"  Output size: {len(response.content)} bytes")
        
        # Save output
        if output_file is None:
            output_file = f"output_{voice_name}.wav"
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"  Output saved: {output_file}")
        
        # Calculate RTF (Real-Time Factor)
        rtf = processing_time / input_duration if input_duration > 0 else 0
        print(f"  RTF: {rtf:.2f}x (lower is better)")
        
        print("  [PASS] Conversion successful")
        return True
        
    except requests.exceptions.Timeout:
        print("  [FAIL] Request timed out")
        return False
    except Exception as e:
        print(f"  [FAIL] Conversion failed: {e}")
        return False


def test_error_handling(base_url: str) -> bool:
    """Test error handling."""
    print("\n[TEST] Error Handling")
    print("-" * 40)
    
    all_passed = True
    
    # Test invalid model
    print("  Testing invalid model name...")
    try:
        response = requests.post(
            f"{base_url}/convert",
            files={"audio": ("test.wav", generate_test_audio(), "audio/wav")},
            data={"voice_name": "nonexistent_model_12345"},
            timeout=30
        )
        if response.status_code == 404:
            print("    [PASS] Returns 404 for invalid model")
        else:
            print(f"    [FAIL] Expected 404, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        all_passed = False
    
    # Test invalid f0_method
    print("  Testing invalid f0_method...")
    try:
        response = requests.post(
            f"{base_url}/convert",
            files={"audio": ("test.wav", generate_test_audio(), "audio/wav")},
            data={"voice_name": "test", "f0_method": "invalid_method"},
            timeout=30
        )
        if response.status_code == 400:
            print("    [PASS] Returns 400 for invalid f0_method")
        else:
            print(f"    [WARN] Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test RVC Inference Server")
    parser.add_argument(
        "--url",
        default="http://localhost:8003",
        help="Base URL of the server (default: http://localhost:8003)"
    )
    parser.add_argument(
        "--voice",
        help="Voice model name to test conversion"
    )
    parser.add_argument(
        "--audio",
        help="Path to input audio file (optional, uses generated audio if not provided)"
    )
    parser.add_argument(
        "--output",
        help="Path to save output audio (default: output_<voice>.wav)"
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (default: 0)"
    )
    parser.add_argument(
        "--method",
        choices=["harvest", "pm"],
        default="harvest",
        help="F0 extraction method (default: harvest)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including error handling"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RVC Inference Server Test")
    print("=" * 50)
    print(f"Server URL: {args.url}")
    
    results = {"passed": 0, "failed": 0}
    
    # Test health
    if test_health(args.url):
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test list voices
    voices = test_list_voices(args.url)
    if voices:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test conversion
    voice_to_test = args.voice
    if not voice_to_test and voices:
        voice_to_test = voices[0]
        print(f"\nNo voice specified, using first available: {voice_to_test}")
    
    if voice_to_test:
        if test_convert(
            args.url,
            voice_to_test,
            args.audio,
            args.pitch,
            args.method,
            args.output
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
    else:
        print("\n[SKIP] No voice models available for conversion test")
    
    # Test error handling
    if args.all:
        if test_error_handling(args.url):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print("=" * 50)
    
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
