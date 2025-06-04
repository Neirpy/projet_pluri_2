from datetime import datetime, timedelta
from controller import Robot, Motion, Speaker, DistanceSensor, Camera

class NaoMotion:
  
   ms = ["Forwards", "Backwards", "TurnLeft40", "TurnRight40", "SideStepLeft", "SideStepRight", "HandWave","Shoot","TaiChi","WipeForehead", "No"]
   motions = {}
  
   def __init__(self, robot):
       self.robot = robot
       self.timestep = int(robot.getBasicTimeStep())
       self._createMotions()
       self.can_anim = True
       self.speaker = robot.getDevice("speaker")
       self.distanceSensor = robot.getDevice("front_sensor")
       self.distanceSensor.enable(64)
       self.cameraTop = robot.getDevice("CameraTop")
       self.cameraTop.enable(64)
       self.cameraTPS = robot.getDevice("cameraTPS")
       self.cameraTPS.enable(64)
       if self.speaker: # Always good practice to check if the device was found
            print("Speaker find")
       else:
            print("Speaker device not found!")
  
   def _createMotions(self):
    for n in self.ms: 
        m3 = self._createSpeeds(n)
        self.motions[n] = m3
      
   def _createSpeeds(self, n):
    with open("../../motions/"+n+".motion") as f:
        header = f.readline()
        lines1 = header
        lines2 = header
        lines3 = header
        line = f.readline()
        while line is not None and line != "":
            t = line[:line.index(",")]
            t = datetime.strptime(t, "%M:%S:%f").time()
            t = timedelta(hours=0, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
            ts1 = f"00:{(t*2).seconds:02d}:{int((t*2).microseconds/1000):03d}"
            ts2 = f"00:{t.seconds:02d}:{int(t.microseconds/1000):03d}"
            ts3 = f"00:{int((t/1.8).seconds):02d}:{int((t/1.8).microseconds/1000):03d}"
            lines1 += ts1+line[line.index(","):] 
            lines2 += line
            lines3 += ts3+line[line.index(","):] 
            line = f.readline()
        with open("../../motions/"+n+"_1.motion", "w") as f:
            f.write(lines1)
        with open("../../motions/"+n+"_2.motion", "w") as f:
            f.write(lines2)
        with open("../../motions/"+n+"_3.motion", "w") as f:
            f.write(lines3)
        return (Motion("../../motions/"+n+"_1.motion"), 
                Motion("../../motions/"+n+"_2.motion"), 
                Motion("../../motions/"+n+"_3.motion"))

   def _checkspeed(self, speed):
        if type(speed) != int: raise TypeError("speed should be an integer")
        if speed < 1 or speed > 3: raise ValueError("speed should be 1, 2, 3")
     
   def _checkdir(self, dir):
        if type(dir) != str: raise TypeError("direction should be a string")
        if dir != "left" and dir != "right": raise ValueError("direction should be left or right")
   
   def _wait(self, duration_ms):
        start_time = self.robot.getTime() * 1000 # Convertir en ms
        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime() * 1000 # Convertir en ms
            if (current_time - start_time) >= duration_ms:
                break
   
   def _applyMotion(self, motion):
        motion.play()
        self.can_anim = False
        # print(motion.getDuration())
        while self.robot.step(self.timestep) != -1:
            if motion.isOver() and motion.getTime() >= motion.getDuration() - 5: 
                # print("---", motion.getTime())
                self.can_anim = True
                return 
                
   def can_get_anim():
       return self.can_anim 
  
  
   def neutral(self, speed):
       self._wait(500)
  
   def change_speed(self, speed):
       self._checkspeed(speed)
       self.speaker.speak(f"Speed set :{speed}",100)
       self._wait(700)
   
   def forward(self, speed):
        if self.distanceSensor.getValue() < 0.40:
            self.cant_move(speed)
            self._wait(500)
        else:
            self._checkspeed(speed) 
            motion = self.motions["Forwards"][speed-1]
            self._applyMotion(motion)
       
   def backward(self, speed):
        self._checkspeed(speed) 
        motion = self.motions["Backwards"][speed-1]
        self._applyMotion(motion)
    
   def turn(self, dir, speed):
       if self.distanceSensor.getValue() < 0.40:
            self.cant_move(speed)
            self._wait(500)
       self._checkdir(dir)
       self._checkspeed(speed)
       if dir=="left": motion = self.motions["TurnLeft40"][speed-1]
       else: motion = self.motions["TurnRight40"][speed-1]
       self._applyMotion(motion)
       
   def sidestep(self, dir, speed):
       self._checkdir(dir)
       self._checkspeed(speed)
       if dir=="left": motion = self.motions["SideStepLeft"][speed-1]
       else: motion = self.motions["SideStepRight"][speed-1]
       self._applyMotion(motion)
   
   def coucou(self,speed):
       self._checkspeed(speed)
       self.speaker.speak("Hello, how are you ?",100)
       motion = self.motions["HandWave"][speed-1]
       self._applyMotion(motion)

   def cant_move(self, speed):
       self._checkspeed(speed)
       self.speaker.speak("Can't move",100)
       motion = self.motions["No"][speed-1]
       self._applyMotion(motion)