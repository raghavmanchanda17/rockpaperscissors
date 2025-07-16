import random
import cv2
import cvzone
from detect import detect_move_with_confidence
import time
 
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

model_path = "./models/rps_cnn.h5"

 
timer = 0
stateResult = False
startGame = False
scores = [0, 0]  
 
while True:
    imgBG = cv2.imread("Resources/BG.png")
    success, img = cap.read()
 
    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]
 

    if startGame:
 
        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
 
            if timer > 3:
                stateResult = True
                timer = 0
 
                # Use CNN to detect move
                try:
                    move_text, confidence = detect_move_with_confidence(model_path, img)
                    playerMove = None
                    if move_text.lower() == "rock":
                        playerMove = 1
                    elif move_text.lower() == "paper":
                        playerMove = 2
                    elif move_text.lower() == "scissors":
                        playerMove = 3
                    print(f"Detected: {move_text} (confidence: {confidence:.2f})")
                except Exception as e:
                    print(f"Detection error: {e}")
                    playerMove = None

                if playerMove is not None:
 
                    randomNumber = random.randint(1, 3)
                    imgAI = cv2.imread(f'Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
 
                    # AI Wins
                    if (playerMove == 1 and randomNumber == 3) or \
                            (playerMove == 2 and randomNumber == 1) or \
                            (playerMove == 3 and randomNumber == 2):
                        scores[0] += 1
 
                    # AI Wins
                    if (playerMove == 3 and randomNumber == 1) or \
                            (playerMove == 1 and randomNumber == 2) or \
                            (playerMove == 2 and randomNumber == 3):
                        scores[1] += 1
 
    imgBG[234:654, 795:1195] = imgScaled
 
    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
 
    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
 
    cv2.imshow("BG", imgBG)
 
    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('q') or key == 27:  # 'q' or ESC to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Game Over! Final Score - AI: {scores[0]}, Player: {scores[1]}")