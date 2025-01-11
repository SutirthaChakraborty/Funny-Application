import easyocr
import pyautogui
from g4f.client import Client
import json
import subprocess
import time
from typing import Tuple, List, Dict, Any
from PIL import Image


class ScreenAutomation:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.client = Client()

    def take_screenshot(self) -> Tuple[Image.Image, str]:
        """Take a screenshot and save it temporarily."""
        screenshot = pyautogui.screenshot()
        screenshot_path = 'screenshot.png'
        screenshot.save(screenshot_path)
        return screenshot, screenshot_path

    def calculate_scaling_factors(self, screenshot: Image.Image) -> Tuple[float, float]:
        """Determine resolutions and calculate scaling factors."""
        screenshot_width, screenshot_height = screenshot.size
        screen_width, screen_height = pyautogui.size()
        width_scaling_factor = screen_width / screenshot_width
        height_scaling_factor = screen_height / screenshot_height
        return width_scaling_factor, height_scaling_factor

    def format_ocr_results(
        self, results: List, width_factor: float, height_factor: float
    ) -> str:
        """Format OCR results with adjusted coordinates."""
        return "\n".join(
            [
                f"Coordinates: [({int(bbox[0][0] * width_factor)}, {int(bbox[0][1] * height_factor)}), "
                f"({int(bbox[2][0] * width_factor)}, {int(bbox[2][1] * height_factor)})], "
                f"Text: '{text}', Confidence: {confidence:.2f}"
                for bbox, text, confidence in results
            ]
        )

    def execute_commands(self, command: str) -> None:
        """Execute a shell command using subprocess."""
        try:
            print(f"[INFO] Executing command: {command}")
            # For macOS 'open' commands, handle them directly without splitting
            if command.startswith('open'):
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
            else:
                # For other commands, split and execute without shell
                cmd_parts = command.split()
                result = subprocess.run(cmd_parts, capture_output=True, text=True)

            if result.returncode == 0:
                print("[INFO] Command executed successfully")
                if result.stdout:
                    print(f"[INFO] Output: {result.stdout}")
            else:
                print(
                    f"[WARNING] Command returned non-zero exit status: {result.returncode}"
                )
                if result.stderr:
                    print(f"[WARNING] Error output: {result.stderr}")

        except Exception as e:
            print(f"[ERROR] Unexpected error executing command: {e}")

    def perform_gui_action(self, action_dict: Dict[str, Any]) -> None:
        """Execute a single PyAutoGUI action."""
        action = action_dict.get("action", "").lower()

        if action == "moveto":
            x = action_dict.get("x", 0)
            y = action_dict.get("y", 0)
            duration = action_dict.get("duration", 0)
            print(f"[INFO] Moving mouse to ({x}, {y}) over {duration} seconds...")
            pyautogui.moveTo(x, y, duration=duration)

        elif action == "click":
            clicks = action_dict.get("clicks", 1)
            button = action_dict.get("button", "left")
            print(f"[INFO] Clicking {button} button {clicks} time(s).")
            pyautogui.click(clicks=clicks, button=button)
            time.sleep(0.5)  # Wait for click effect

        elif action == "typewrite":
            text = action_dict.get("text", "")
            interval = action_dict.get("interval", 0.0)
            print(f"[INFO] Typing text: {text}")
            pyautogui.typewrite(text, interval=interval)

        elif action == "hotkey":
            key = action_dict.get("key", "")
            modifier = action_dict.get("modifier", "")
            if modifier and key:
                print(f"[INFO] Pressing hotkey combination: {modifier}+{key}")
                pyautogui.keyDown(modifier)
                pyautogui.press(key)
                pyautogui.keyUp(modifier)
            elif key:
                print(f"[INFO] Pressing key: {key}")
                pyautogui.press(key)
            time.sleep(0.5)  # Wait for hotkey effect

        elif action == "wait":
            duration = action_dict.get("duration", 1)
            print(f"[INFO] Waiting for {duration} seconds...")
            time.sleep(duration)

        else:
            print(f"[WARNING] Unrecognized pyautogui action: {action_dict}")

    def refresh_screen_state(self) -> Tuple[List, float, float]:
        """Take a new screenshot and refresh OCR results."""
        screenshot, screenshot_path = self.take_screenshot()
        width_factor, height_factor = self.calculate_scaling_factors(screenshot)
        results = self.reader.readtext(screenshot_path)
        return results, width_factor, height_factor

    def clean_json_response(self, response_content: str) -> str:
        """Clean GPT response to extract valid JSON content."""
        # Remove backticks and json indicators
        content = response_content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]

        # Remove any leading/trailing whitespace after cleaning
        content = content.strip()
        return content

    def process_gpt_response(self, response_content: str) -> List[Dict[str, Any]]:
        """Process GPT response and extract sequence of actions."""
        try:
            # Clean the response first
            cleaned_content = self.clean_json_response(response_content)
            print("\nCleaned JSON content:")
            print(cleaned_content)

            # Parse the cleaned JSON
            instructions = json.loads(cleaned_content)
            return instructions.get("sequence", [])
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {e}")
            print(f"[DEBUG] Attempted to parse content: {cleaned_content}")
            return []
        except Exception as e:
            print(f"[ERROR] Unexpected error processing response: {e}")
            return []

    def run(self):
        """Main execution loop."""
        while True:
            # Take screenshot and process it
            screenshot, screenshot_path = self.take_screenshot()
            width_factor, height_factor = self.calculate_scaling_factors(screenshot)

            # Perform OCR
            results = self.reader.readtext(screenshot_path)
            ocr_results_formatted = self.format_ocr_results(
                results, width_factor, height_factor
            )

            print("OCR Results:")
            print(ocr_results_formatted)

            # Get user instruction
            user_instruction = input(
                "\nEnter your instruction for the GPT model (or type 'exit' to quit): "
            )
            if user_instruction.lower() == 'exit':
                print("Exiting program.")
                break

            # Prepare instruction for GPT
            instruction = f"""Instruction: {user_instruction}.
            INSTRUCTIONS: (I am using a MAC PC)
            1. Return ONLY valid JSON with a single key "sequence" containing a list of action objects.
            2. Follow these guidelines for creating sequences:
               - Start with shell commands for file/system operations
               - Use GUI actions for interactive tasks
               - Include proper wait times between actions
               - Add screen updates before critical GUI interactions
               - Handle file operations and GUI interactions systematically

            3. Common patterns to follow:
               A. For file creation and editing:
                  - Create file with shell command
                  - Open in appropriate application
                  - Wait for application to load
                  - Perform typing/editing
                  - Save and close

               B. For browser operations:
                  - Use direct URL opening when possible
                  - Include waits after browser launch
                  - Add fallback GUI actions if needed

               C. For system operations:
                  - Prefer shell commands when available
                  - Add verification steps when needed
                  - Include cleanup actions if necessary

            4. Available actions:
               Shell actions:
               {{"type": "shell", "command": "command string"}}

               GUI actions:
               {{"type": "gui", "action": "moveTo", "x": int, "y": int, "duration": float}}
               {{"type": "gui", "action": "click", "clicks": int, "button": "left|right"}}
               {{"type": "gui", "action": "typewrite", "text": string, "interval": float}}
               {{"type": "gui", "action": "hotkey", "modifier": string, "key": string}}
               {{"type": "gui", "action": "wait", "duration": float}}

            5. Example sequences:
               A. Creating and editing a text file:
               {{
                 "sequence": [
                   {{"type": "shell", "command": "touch myfile.txt"}},
                   {{"type": "shell", "command": "open -a TextEdit myfile.txt"}},
                   {{"type": "gui", "action": "wait", "duration": 2.0}},
                   {{"type": "gui", "action": "typewrite", "text": "Line 1\\nLine 2\\nLine 3"}},
                   {{"type": "gui", "action": "hotkey", "modifier": "command", "key": "s"}},
                   {{"type": "gui", "action": "wait", "duration": 0.5}},
                   {{"type": "gui", "action": "hotkey", "modifier": "command", "key": "w"}}
                 ]
               }}

               B. Opening URL in browser:
               {{
                 "sequence": [
                   {{"type": "shell", "command": "open -a 'Google Chrome' 'https://www.example.com'"}},
                   {{"type": "gui", "action": "wait", "duration": 2.0}},
                   {{"type": "gui", "action": "hotkey", "modifier": "command", "key": "l"}},
                   {{"type": "gui", "action": "typewrite", "text": "https://www.example.com"}},
                   {{"type": "gui", "action": "hotkey", "key": "return"}}
                 ]
               }}

            Here are the detected texts and their coordinates:\n{ocr_results_formatted}\n\n

            Ensure the sequence is complete and includes all necessary waiting periods and error handling steps."""

            # Get GPT response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": instruction}],
                web_search=False,
            )

            print("\nGPT Response:")
            print(response.choices[0].message.content)

            # Process GPT response
            sequence = self.process_gpt_response(response.choices[0].message.content)

            # Execute sequence of actions
            for action in sequence:
                action_type = action.get("type")

                if action_type == "shell":
                    command = action.get("command")
                    if command:
                        self.execute_commands(command)
                        time.sleep(0.5)  # Brief pause after shell commands

                elif action_type == "gui":
                    # Check if screen update is needed
                    if action.get("needs_screen_update", False):
                        _, _, _ = self.refresh_screen_state()

                    # Extract and execute the GUI action
                    gui_action = {
                        k: v
                        for k, v in action.items()
                        if k not in ["type", "needs_screen_update"]
                    }
                    self.perform_gui_action(gui_action)

            print("\n[INFO] Task completed. Ready for the next instruction.")


if __name__ == "__main__":
    automation = ScreenAutomation()
    automation.run()
