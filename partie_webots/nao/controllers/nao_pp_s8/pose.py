import mediapipe as mp
import cv2
import requests

class Pose:

  def __init__(self, min_detection_confidence, min_tracking_confidence, 
               model_complexity, video_source):
      self.mp_pose = mp.solutions.pose
      self.cap = cv2.VideoCapture(video_source)
      self.min_detection_confidence = min_detection_confidence
      self.min_tracking_confidence = min_tracking_confidence
      self.model_complexity = model_complexity
      self.video_source = video_source
      self.url = "http://127.0.0.1:8000/predict"
 
  def _toVector(self, landmarks):
    if landmarks is None: return []
    line = []
    for i,l in enumerate(landmarks.landmark):
       line.append(str(l.x))
       line.append(str(l.y))
    return line
    
  
  def getPose(self):
    vec = [] # Initialise vec à vide

    # --- Partie 1: Capture et traitement de la pose avec MediaPipe ---
    try:
        with self.mp_pose.Pose(min_detection_confidence=self.min_detection_confidence,
                                 min_tracking_confidence=self.min_tracking_confidence,
                                 model_complexity=self.model_complexity) as pose:
            opened, image = self.cap.read()
            if not opened:
                print(f"Error: Could not read frame from video source: {self.video_source}")
                return None # Retourne None si la capture de la caméra échoue

            # Convertir l'image BGR en RGB avant de la traiter avec MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Convertir les landmarks de pose en votre format vectoriel
            vec = self._toVector(results.pose_landmarks)

    except Exception as e:
        print(f"An error occurred during pose processing: {e}")
        return None # Retourne None si une erreur se produit pendant le traitement de la pose

    # --- Partie 2: Envoi des données 'vec' via une requête POST ---
    try:
        # Le 'payload' est le dictionnaire Python qui sera converti en JSON
        # et envoyé dans le corps de la requête POST.
        # J'ai encapsulé 'vec' sous une clé 'pose_data', vous pouvez adapter le nom.
        payload = {"values": vec}

        # Envoyer une requête POST à l'URL.
        # Le paramètre 'json=payload' gère la sérialisation en JSON et l'en-tête Content-Type.
        response = requests.post(self.url, json=payload)

        # Vérifier si la requête a réussi (code de statut 200)
        if response.status_code == 200:
            #print("POST request successful!")
            # Si la réponse du serveur est JSON, vous pouvez la parser directement:
            data = response.json()
            return data # Retourne les données JSON reçues du serveur après le POST
        else:
            print(f"POST request failed with status code: {response.status_code}")
            print("Response content (if any):")
            print(response.text)
            return None # Retourne None en cas d'échec de la requête POST

    except requests.exceptions.ConnectionError as e:
        print(f"Error: Could not connect to the server at {self.url}") # Correction: utiliser self.url
        print("Please ensure the server is running and accessible.")
        print(f"Details: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Error: The POST request to {self.url} timed out.") # Correction: utiliser self.url
        print(f"Details: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An unexpected error occurred during the POST request: {e}")
        return None
    except Exception as e:
        # Capture d'autres erreurs potentielles (par exemple, si la réponse n'est pas JSON valide)
        print(f"An unexpected error occurred after sending the request: {e}")
        return None
