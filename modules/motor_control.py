"""
Motor Control Module for Wall-E Companion Robot
Owner: Dilmurod

This module handles all physical movement according to the API contract.
Implements tank-style locomotion with independent left and right track control.
"""

import logging
import time
from typing import Optional

# Hardware-specific imports (uncomment when running on Jetson)
# import Jetson.GPIO as GPIO
# import RPi.GPIO as GPIO  # Alternative for Raspberry Pi compatibility

logger = logging.getLogger(__name__)

# GPIO Pin definitions (adjust based on your hardware setup)
LEFT_MOTOR_PIN1 = 18   # Left motor direction pin 1
LEFT_MOTOR_PIN2 = 19   # Left motor direction pin 2
LEFT_MOTOR_PWM = 12    # Left motor PWM (speed) pin

RIGHT_MOTOR_PIN1 = 20  # Right motor direction pin 1
RIGHT_MOTOR_PIN2 = 21  # Right motor direction pin 2
RIGHT_MOTOR_PWM = 13   # Right motor PWM (speed) pin

# Global variables
_gpio_initialized = False
_left_pwm = None
_right_pwm = None

def setup() -> None:
    """
    Initializes all GPIO pins for motor control.
    Must be called before any other motor functions.
    Raises RuntimeError if GPIO initialization fails.
    """
    global _gpio_initialized, _left_pwm, _right_pwm
    
    logger.info("Initializing motor control GPIO pins...")
    
    try:
        # TODO: Uncomment and configure when running on actual hardware
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setwarnings(False)
        
        # # Set up left motor pins
        # GPIO.setup(LEFT_MOTOR_PIN1, GPIO.OUT)
        # GPIO.setup(LEFT_MOTOR_PIN2, GPIO.OUT)
        # GPIO.setup(LEFT_MOTOR_PWM, GPIO.OUT)
        
        # # Set up right motor pins
        # GPIO.setup(RIGHT_MOTOR_PIN1, GPIO.OUT)
        # GPIO.setup(RIGHT_MOTOR_PIN2, GPIO.OUT)
        # GPIO.setup(RIGHT_MOTOR_PWM, GPIO.OUT)
        
        # # Initialize PWM for speed control (1000 Hz frequency)
        # _left_pwm = GPIO.PWM(LEFT_MOTOR_PWM, 1000)
        # _right_pwm = GPIO.PWM(RIGHT_MOTOR_PWM, 1000)
        
        # # Start PWM with 0% duty cycle (stopped)
        # _left_pwm.start(0)
        # _right_pwm.start(0)
        
        # # Ensure motors are stopped initially
        # GPIO.output(LEFT_MOTOR_PIN1, GPIO.LOW)
        # GPIO.output(LEFT_MOTOR_PIN2, GPIO.LOW)
        # GPIO.output(RIGHT_MOTOR_PIN1, GPIO.LOW)
        # GPIO.output(RIGHT_MOTOR_PIN2, GPIO.LOW)
        
        _gpio_initialized = True
        logger.info("Motor control initialized successfully")
        
        # TEMPORARY: Simulation mode for development
        logger.warning("Running in SIMULATION mode - no actual GPIO control")
        _gpio_initialized = True
        
    except Exception as e:
        logger.error(f"Failed to initialize motor control: {e}")
        raise RuntimeError(f"Motor control initialization failed: {e}")

def move(left_speed: int, right_speed: int) -> None:
    """
    Sets the speed of each track independently.
    
    Args:
        left_speed: Speed for left track (-100 to 100)
        right_speed: Speed for right track (-100 to 100)
        
    Raises:
        ValueError: If speeds are outside valid range
        RuntimeError: If GPIO is not initialized
    """
    if not _gpio_initialized:
        raise RuntimeError("Motor control not initialized. Call setup() first.")
    
    # Validate speed ranges
    if not (-100 <= left_speed <= 100):
        raise ValueError(f"Left speed {left_speed} out of range [-100, 100]")
    if not (-100 <= right_speed <= 100):
        raise ValueError(f"Right speed {right_speed} out of range [-100, 100]")
    
    logger.debug(f"Setting motor speeds: left={left_speed}, right={right_speed}")
    
    # TODO: Uncomment when running on actual hardware
    # _set_motor_speed("left", left_speed)
    # _set_motor_speed("right", right_speed)
    
    # TEMPORARY: Simulation mode
    if left_speed != 0 or right_speed != 0:
        logger.info(f"SIMULATION: Moving - Left: {left_speed}%, Right: {right_speed}%")
    else:
        logger.info("SIMULATION: Motors stopped")

def _set_motor_speed(motor: str, speed: int) -> None:
    """
    Internal function to set individual motor speed and direction.
    
    Args:
        motor: "left" or "right"
        speed: Speed from -100 to 100
    """
    # TODO: Implement actual GPIO control
    # This is a template - implement based on your motor driver
    
    if motor == "left":
        pin1, pin2, pwm = LEFT_MOTOR_PIN1, LEFT_MOTOR_PIN2, _left_pwm
    else:
        pin1, pin2, pwm = RIGHT_MOTOR_PIN1, RIGHT_MOTOR_PIN2, _right_pwm
    
    # Convert speed to PWM duty cycle (0-100)
    duty_cycle = abs(speed)
    
    if speed > 0:
        # Forward direction
        # GPIO.output(pin1, GPIO.HIGH)
        # GPIO.output(pin2, GPIO.LOW)
        pass
    elif speed < 0:
        # Reverse direction
        # GPIO.output(pin1, GPIO.LOW)
        # GPIO.output(pin2, GPIO.HIGH)
        pass
    else:
        # Stop
        # GPIO.output(pin1, GPIO.LOW)
        # GPIO.output(pin2, GPIO.LOW)
        pass
    
    # Set PWM duty cycle
    # pwm.ChangeDutyCycle(duty_cycle)

def stop() -> None:
    """
    Immediately stops all motors.
    Safe to call multiple times.
    """
    if not _gpio_initialized:
        logger.warning("Motor control not initialized, cannot stop motors")
        return
    
    logger.info("Stopping all motors")
    
    # TODO: Uncomment when running on actual hardware
    # GPIO.output(LEFT_MOTOR_PIN1, GPIO.LOW)
    # GPIO.output(LEFT_MOTOR_PIN2, GPIO.LOW)
    # GPIO.output(RIGHT_MOTOR_PIN1, GPIO.LOW)
    # GPIO.output(RIGHT_MOTOR_PIN2, GPIO.LOW)
    
    # if _left_pwm:
    #     _left_pwm.ChangeDutyCycle(0)
    # if _right_pwm:
    #     _right_pwm.ChangeDutyCycle(0)
    
    # TEMPORARY: Simulation mode
    logger.info("SIMULATION: All motors stopped")

def cleanup() -> None:
    """
    Releases all GPIO pins safely when the program exits.
    Should be called in exception handlers and at program termination.
    """
    global _gpio_initialized, _left_pwm, _right_pwm
    
    if not _gpio_initialized:
        return
    
    logger.info("Cleaning up motor control...")
    
    try:
        # Stop all motors first
        stop()
        
        # TODO: Uncomment when running on actual hardware
        # # Stop PWM
        # if _left_pwm:
        #     _left_pwm.stop()
        # if _right_pwm:
        #     _right_pwm.stop()
        
        # # Clean up GPIO
        # GPIO.cleanup()
        
        _gpio_initialized = False
        _left_pwm = None
        _right_pwm = None
        
        logger.info("Motor control cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during motor control cleanup: {e}")

# Emergency stop function (can be called from anywhere)
def emergency_stop() -> None:
    """Emergency stop - immediately stops all motors regardless of state."""
    logger.warning("EMERGENCY STOP activated!")
    try:
        stop()
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")

# Test functions for development
def test_motors() -> None:
    """Test function to verify motor control is working."""
    if not _gpio_initialized:
        logger.error("Cannot test motors - not initialized")
        return
    
    logger.info("Testing motors...")
    
    # Test forward movement
    logger.info("Testing forward movement...")
    move(50, 50)
    time.sleep(2)
    
    # Test turning
    logger.info("Testing left turn...")
    move(30, -30)
    time.sleep(1)
    
    logger.info("Testing right turn...")
    move(-30, 30)
    time.sleep(1)
    
    # Stop
    logger.info("Stopping motors...")
    stop()
    
    logger.info("Motor test complete")

if __name__ == "__main__":
    # Test the module independently
    logging.basicConfig(level=logging.INFO)
    
    try:
        setup()
        test_motors()
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    finally:
        cleanup()
