import qi
import argparse
import sys
import time
import requests # Import the requests library
import json     # Import the json library

class RobotController:
    def __init__(self, session, command_url):
        self.session = session
        self.command_url = command_url 
        self.motion_service = session.service("ALMotion")
        self.posture_service = session.service("ALRobotPosture")
        self.asr_service = session.service("ALAnimatedSpeech")
        self.current_frequency = 0.75
        self.configuration = {"bodyLanguageMode":"disabled"} 

    def get_commands(self):
        """
        Sends a GET request to fetch commands from the specified URL.
        Expected response: JSON array like ["action", "speed"].
        """
        try:
            response = requests.get(self.command_url)

            if response.status_code == 200:
                print("GET request successful!")
                try:
                    data = response.json()
                    print("Server response (GET): {}".format(data))
                    return data
                except ValueError:
                    print("GET Response content is not valid JSON.")
                    print(response.text)
                    return None
            else:
                print("GET request failed with status code: {}".format(response.status_code))
                print("GET Response content (if any):")
                print(response.text)
                return None

        except requests.exceptions.ConnectionError as e:
            print("Error: Could not connect to the GET server at {}".format(self.command_url))
            print("Please ensure the server is running and accessible.")
            print("Details: {}".format(e))
            return None
        except requests.exceptions.Timeout as e:
            print("Error: The GET request to {} timed out.".format(self.command_url))
            print("Details: {}".format(e))
            return None
        except requests.exceptions.RequestException as e:
            print("An unexpected error occurred during the GET request: {}".format(e))
            return None
        except Exception as e:
            print("An unexpected error occurred after sending the GET request: {}".format(e))
            return None

    def execute_robot_action(self, action, speed_level):
        """
        Executes robot movement based on the received action and speed.
        """
        x, y, theta = 0.0, 0.0, 0.0

        # Ne change plus la vitesse, il faudrait faire un setMoveSpeed
        if speed_level == "LENT" and self.current_frequency != 0.5 :
            self.current_frequency -= 0.25
        if speed_level == "RAPIDE" and self.current_frequency != 1.0 :
            self.current_frequency += 0.25
        

        print("Executing action: {} with speed level: {} (frequency: {})".format(action, speed_level, self.current_frequency))

        if action == "AVANT":
            x = 0.2
        elif action == "ARRIERE":
            x = -0.2
        elif action == "DROITE":
            y = -0.1
        elif action == "GAUCHE":
            y = 0.1
        elif action == "TOURNER_DROITE":
            theta = -0.4 # Radians, adjust as needed
        elif action == "TOURNER_GAUCHE":
            theta = 0.4 # Radians, adjust as needed
        elif action == "SURPRISE":
            self.asr_service.say("^start(animations/Stand/Emotions/Positive/Happy_4) Hello how are you ? ^wait(animations/Stand/Emotions/Positive/Happy_4)", self.configuration)
            return # Don't move if it's a speech command
        elif action == "NEUTRE":
            self.motion_service.stopMove()
            return # Don't move, just stop
        print("execute action")
        # Execute movement
        self.motion_service.moveTo(x, y, theta)
        self.motion_service.waitUntilMoveIsFinished()
        self.motion_service.stopMove()


    def main(self):
        """
        This example continuously fetches commands and controls the robot.
        """
        self.motion_service.stopMove()
        self.motion_service.wakeUp()
        self.posture_service.goToPosture("StandInit", 0.5)

        print("Robot ready. Starting command loop...")

        while True: # Loop indefinitely to continuously get and execute commands
            # 1. Get commands from the server
            commands = self.get_commands()

            if commands and isinstance(commands, list) and len(commands) == 2:
                action = commands[1]
                speed_level = commands[0]

                # 2. Execute robot action based on commands
                self.execute_robot_action(action, speed_level)
            else:
                print("Failed to get valid commands. Robot will remain still for a moment.")
                self.motion_service.stopMove()

            self.motion_service.waitUntilMoveIsFinished()

            #time.sleep(5) # Pause before the next cycle of fetching commands

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #192.168.43.157
    parser.add_argument("--ip", type=str, default="192.168.43.157",
                        help="Robot IP address. On robot or Local Naoqi: use '192.168.43.157'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    parser.add_argument("--command_url", type=str, default="http://localhost:8000/get_prediction",
                        help="URL of the server endpoint to GET commands from.")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    controller = RobotController(session, args.command_url)
    controller.main()
