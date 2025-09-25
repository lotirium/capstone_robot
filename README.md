# Project Wall-E: The AI Companion Robot

This is the official repository for our Capstone Design project to build an AI-powered companion robot using a Jetson Orin NX. Our goal is to create an interactive robot that can follow people, respond to voice commands, and provide intelligent conversation through advanced AI capabilities.

## Project Overview

Wall-E is designed to be a friendly companion robot that combines:
- **Computer Vision**: Person detection and obstacle avoidance using OAK-D camera
- **Natural Language Processing**: Voice recognition and AI-powered responses
- **Autonomous Movement**: Tank-style locomotion with independent track control
- **Real-time Processing**: All running on NVIDIA Jetson Orin NX for edge AI computing

## Hardware Components
- NVIDIA Jetson Orin NX (main computing unit)
- OAK-D Camera (depth perception and computer vision)
- Tank chassis with independent track motors
- Speakers and microphone for audio interaction
- Various sensors for environmental awareness

## Team Members
- **Dilmurod**: Motor Control & Hardware Integration
- **Sabera**: Computer Vision & Object Detection  
- **Boymirzo**: Audio Processing & AI Integration

---

## Module API Contract
This document defines the official functions our software modules will use to communicate with `main.py`.

### `modules/motor_control.py` (Owner: Dilmurod)
*This module handles all physical movement.*

- **`setup() -> None:`**
  - Initializes all GPIO pins for motor control.
  - Must be called before any other motor functions.
  - Raises `RuntimeError` if GPIO initialization fails.

- **`move(left_speed: int, right_speed: int) -> None:`**
  - Sets the speed of each track independently. 
  - Speed is an integer from -100 (full reverse) to 100 (full forward).
  - `left_speed`: Speed for left track (-100 to 100)
  - `right_speed`: Speed for right track (-100 to 100)
  - Raises `ValueError` if speeds are outside valid range.

- **`stop() -> None:`**
  - Immediately stops all motors.
  - Safe to call multiple times.

- **`cleanup() -> None:`**
  - Releases all GPIO pins safely when the program exits.
  - Should be called in exception handlers and at program termination.

---

### `modules/vision.py` (Owner: Sabera)
*This module handles all input from the OAK-D camera.*

- **`setup() -> None:`**
  - Initializes the OAK-D camera and computer vision pipeline.
  - Must be called before any other vision functions.
  - Raises `RuntimeError` if camera initialization fails.

- **`get_latest_frame() -> numpy.ndarray:`**
  - Returns the latest color image frame as a NumPy array.
  - Format: BGR color image (OpenCV standard)
  - Shape: (height, width, 3)
  - Returns `None` if no frame is available.

- **`is_person_detected() -> bool:`**
  - Returns `True` if a person is detected in the current frame, otherwise `False`.
  - Uses YOLO or similar object detection model.
  - Updates automatically with each new frame.

- **`get_obstacle_distance() -> float:`**
  - Returns the distance in meters to the nearest obstacle directly in front of the robot.
  - Uses depth information from OAK-D camera.
  - Returns `float('inf')` if no obstacle is detected within range.
  - Range: 0.5 to 10.0 meters (camera limitations).

- **`cleanup() -> None:`**
  - Properly closes camera connections and releases resources.

---

### `modules/audio.py` (Owner: Boymirzo)
*This module handles all audio input and output.*

- **`setup() -> None:`**
  - Initializes audio hardware (microphone and speakers).
  - Sets up speech recognition and text-to-speech engines.
  - Must be called before any other audio functions.
  - Raises `RuntimeError` if audio hardware initialization fails.

- **`listen_and_transcribe() -> str:`**
  - Listens for speech, transcribes it, and returns the recognized text as a string.
  - Blocks until speech is detected and processed.
  - Returns empty string `""` if no speech is recognized.
  - Timeout: 5 seconds of silence before returning.

- **`speak(text: str) -> None:`**
  - Takes a string of text and speaks it out loud using text-to-speech.
  - `text`: The message to be spoken
  - Non-blocking: returns immediately while speech continues in background.
  - Raises `ValueError` if text is empty or None.

- **`get_intelligent_response(prompt: str) -> str:`**
  - Sends a text prompt to the LLM and returns the AI's response.
  - `prompt`: User's question or statement to send to AI
  - Returns the AI's response as a string.
  - Handles API rate limiting and network errors gracefully.
  - Returns error message if LLM is unavailable.

- **`cleanup() -> None:`**
  - Stops any ongoing speech and releases audio resources.

---

## Main Program Structure

The `main.py` file will orchestrate all modules:

```python
# Example usage of the API contract
import modules.motor_control as motor
import modules.vision as vision  
import modules.audio as audio

def main():
    # Initialize all modules
    motor.setup()
    vision.setup()
    audio.setup()
    
    try:
        while True:
            # Check for person and follow
            if vision.is_person_detected():
                # Simple following logic
                motor.move(50, 50)  # Move forward
            else:
                motor.stop()
            
            # Check for obstacles
            distance = vision.get_obstacle_distance()
            if distance < 1.0:  # Too close to obstacle
                motor.stop()
            
            # Listen for voice commands
            command = audio.listen_and_transcribe()
            if command:
                response = audio.get_intelligent_response(command)
                audio.speak(response)
                
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up all modules
        motor.cleanup()
        vision.cleanup()
        audio.cleanup()

if __name__ == "__main__":
    main()
```

## Development Guidelines

1. **Error Handling**: All modules must handle errors gracefully and provide meaningful error messages.
2. **Documentation**: Each function should include docstrings with parameter descriptions and return values.
3. **Testing**: Create unit tests for each module's functions.
4. **Dependencies**: Document all required Python packages in `requirements.txt`.
5. **Hardware Safety**: Motor control must include emergency stop functionality.

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Connect hardware components according to wiring diagram
4. Run the main program: `python main.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
