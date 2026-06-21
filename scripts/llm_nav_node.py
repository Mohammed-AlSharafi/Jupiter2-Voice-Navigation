#!/usr/bin/env python3
import json

import actionlib
import requests
import rospy
from geometry_msgs.msg import Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String


class LLMNavNode:
    def __init__(self):
        rospy.init_node("llm_nav_node")

        self.input_topic = rospy.get_param("~input_topic", "/llm_input")
        self.api_key = rospy.get_param("~api_key", "")
        self.openai_base = rospy.get_param("~openai_base", "https://openrouter.ai/api/v1")
        self.model = rospy.get_param("~model", "openrouter/free")
        self.nav_timeout = float(rospy.get_param("~navigation_timeout", 60.0))
        self.locations = rospy.get_param("~locations", {})

        if not self.api_key:
            rospy.logfatal("[llm_nav] ~api_key is not set. Shutting down.")
            rospy.signal_shutdown("Missing API key")
            return
        if not self.locations:
            rospy.logfatal("[llm_nav] No locations loaded. Shutting down.")
            rospy.signal_shutdown("Missing locations")
            return

        rospy.loginfo(
            "[llm_nav] Loaded %d location(s): %s",
            len(self.locations),
            list(self.locations.keys()),
        )
        rospy.loginfo("[llm_nav] Using model: %s", self.model)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        rospy.loginfo("[llm_nav] Connecting to move_base action server…")
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        if not self.move_base.wait_for_server(rospy.Duration(15.0)):
            rospy.logfatal("[llm_nav] move_base not available. Shutting down.")
            rospy.signal_shutdown("Navigation server not available")
            return
        rospy.loginfo("[llm_nav] Connected to move_base.")

        self.reply_pub = rospy.Publisher("/llm_reply", String, queue_size=10)
        self.sub = rospy.Subscriber(
            self.input_topic, String, self.text_callback, queue_size=5
        )
        rospy.loginfo("[llm_nav] Subscribed to '%s'. Ready.", self.input_topic)

    def navigate_to_location(self, location_name):
        if location_name not in self.locations:
            msg = f"Unknown location '{location_name}'. Available: {list(self.locations.keys())}"
            rospy.logerr("[llm_nav] %s", msg)
            return msg

        loc = self.locations[location_name]
        x, y = float(loc.get("x", 0.0)), float(loc.get("y", 0.0))
        z, w = float(loc.get("z", 0.0)), float(loc.get("w", 1.0))
        rospy.loginfo(
            "[llm_nav] Navigating to '%s' (x=%.2f, y=%.2f, z=%.2f, w=%.2f)…",
            location_name, x, y, z, w,
        )

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(x=x, y=y)
        goal.target_pose.pose.orientation = Quaternion(z=z, w=w)

        self.move_base.send_goal(goal)
        if not self.move_base.wait_for_result(rospy.Duration(self.nav_timeout)):
            self.move_base.cancel_goal()
            return f"Navigation to '{location_name}' timed out."

        if self.move_base.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[llm_nav] Arrived at '%s'.", location_name)
            return f"Successfully navigated to '{location_name}'."

        rospy.logwarn(
            "[llm_nav] Navigation to '%s' failed (state %d).",
            location_name,
            self.move_base.get_state(),
        )
        return f"Navigation to '{location_name}' failed."

    def _tool_schema(self):
        location_names = list(self.locations.keys())
        return [{
            "type": "function",
            "function": {
                "name": "navigate_to_location",
                "description": (
                    "Send the robot to a named location. "
                    f"Valid locations: {location_names}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_name": {
                            "type": "string",
                            "enum": location_names,
                            "description": "Exact name of the destination location.",
                        }
                    },
                    "required": ["location_name"],
                },
            },
        }]

    def _request(self, user_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful robot assistant. "
                    "Use the navigation tool when the user asks the robot to go "
                    f"somewhere. Valid locations: {list(self.locations.keys())}. "
                    "Match slight variations to the closest valid name before "
                    "calling the tool. If the location is clearly not in the "
                    "list, do not navigate."
                ),
            },
            {"role": "user", "content": user_text},
        ]

        url = f"{self.openai_base.rstrip('/')}/chat/completions"
        try:
            resp = requests.post(
                url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": self._tool_schema(),
                    "tool_choice": "auto",
                },
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            rospy.logerr("[llm_nav] Request failed: %s", e)
            return None

        return resp.json()["choices"][0]["message"]

    def call_llm(self, user_text):
        message = self._request(user_text)
        if message is None:
            return "Error contacting LLM."

        if not message.get("tool_calls"):
            return message.get("content") or "Done."

        for tc in message["tool_calls"]:
            if tc["function"]["name"] != "navigate_to_location":
                return f"Unknown tool '{tc['function']['name']}'."
            args = json.loads(tc["function"]["arguments"])
            return self.navigate_to_location(args.get("location_name", ""))

        return "Done."

    def text_callback(self, msg):
        user_text = msg.data.strip()
        if not user_text:
            return

        rospy.loginfo("[llm_nav] Received: %s", user_text)
        reply = self.call_llm(user_text)
        rospy.loginfo("[llm_nav] LLM reply: %s", reply)
        self.reply_pub.publish(String(data=reply))


if __name__ == "__main__":
    try:
        LLMNavNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
