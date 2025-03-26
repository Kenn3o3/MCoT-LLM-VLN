from typing import Tuple, List
from openai import OpenAI
import numpy as np

class InferenceManager:
    """Handles inference logic for the VLN agent."""
    def __init__(self, vln_model, image_history_manager, client: OpenAI):
        self.vln_model = vln_model
        self.image_history_manager = image_history_manager
        self.client = client

    def run_inference(self, instruction: str, history_actions: List[str]) -> Tuple[str, str, str, List[str]]:
        full_history = self.image_history_manager.get_full_history()
        scene_descs = []
        content = []
        if len(full_history) == 1:
            content.append({"type": "text", "text": "The navigation task now beginning."})
        # Process historical observations
        for i in range(len(full_history) - 1):
            rgb, depth, subtask_idx, action, pos = full_history[i]
            steps_ago = len(full_history) - 1 - i
            rgb_url = self.vln_model._encode_image(rgb)
            depth_colored = self.vln_model._colorize_depth(depth)
            depth_url = self.vln_model._encode_image(depth_colored)
            scene_desc = self.vln_model._generate_scene_description(rgb_url, depth_url, self.vln_model.subtasks[subtask_idx])
            scene_descs.append(scene_desc)

            content.extend([
                {"type": "text", "text": f"Observation from {steps_ago} steps ago:"},
                {"type": "image_url", "image_url": {"url": rgb_url}},
                {"type": "text", "text": scene_desc},
                {"type": "text", "text": f"Then, action taken: {action}"}
            ])

        # Process current observation
        rgb, depth, subtask_idx, action, pos = full_history[-1]
        rgb_url = self.vln_model._encode_image(rgb)
        depth_colored = self.vln_model._colorize_depth(depth)
        depth_url = self.vln_model._encode_image(depth_colored)
        scene_desc = self.vln_model._generate_scene_description(rgb_url, depth_url, self.vln_model.subtasks[subtask_idx])
        scene_descs.append(scene_desc)

        content.extend([
            {"type": "text", "text": "Current observation:"},
            {"type": "image_url", "image_url": {"url": rgb_url}},
            {"type": "text", "text": scene_desc}
        ])

        # Reinforced instruction in the user prompt
        content.append({
            "type": "text",
            "text": f"Based on the instruction: '{instruction}', and the sequence above, decide the next action. "
                    "You MUST respond in this exact format:\n"
                    "Subtasks Completed: [Yes/No]\n"
                    "Reasoning: [your step-by-step reasoning]\n"
                    "Next Action: [Must be one of: Move forward, Turn right, Turn left]\n"
                    "Do not include any additional text outside this format."
        })

        # Add system message to enforce format
        messages = [
            {
                "role": "system",
                "content": "You are a navigation assistant. Your response must be in the following format:\n"
                        "Subtasks Completed: [Yes/No]\n"
                        "Reasoning: [your step-by-step reasoning]\n"
                        "Next Action: [Must be one of: Move forward, Turn right, Turn left]\n"
                        "Do not include any additional text outside this format."
            },
            {
                "role": "user",
                "content": content
            }
        ]

        # Send to VLM
        response = self.client.chat.completions.create(
            model="qwen-vl-plus",
            messages=messages,
            max_tokens=1000
        )
        raw_response = response.choices[0].message.content
        action = self.vln_model._parse_response(raw_response)
        prompt = "\n".join([item["text"] for item in content if item["type"] == "text"])
        return action, prompt, raw_response, scene_descs