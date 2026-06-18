#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

BANNER = """
╔══════════════════════════════════════╗
║     Room Guide — Navigate Mode       ║
╠══════════════════════════════════════╣
║  Type a prompt and press Enter       ║
║  Ctrl-C to quit                      ║
╚══════════════════════════════════════╝
"""


class PromptNode:
    def __init__(self):
        rospy.init_node("prompt_node")

        self.pub = rospy.Publisher("/llm_input", String, queue_size=10)
        self.sub = rospy.Subscriber("/llm_reply", String, self._reply_cb)

        # Give publisher time to connect
        rospy.sleep(0.5)

        print(BANNER)

    def _reply_cb(self, msg: String):
        print(f"\n[Robot] {msg.data}\n> ", end="", flush=True)

    def run(self):
        try:
            while not rospy.is_shutdown():
                try:
                    text = input("> ").strip()
                except EOFError:
                    break

                if not text:
                    continue

                if text.lower() in ("q", "quit", "exit"):
                    break

                self.pub.publish(String(data=text))
        except KeyboardInterrupt:
            pass
        print("\n[prompt] Goodbye.")


if __name__ == "__main__":
    try:
        PromptNode().run()
    except rospy.ROSInterruptException:
        pass
