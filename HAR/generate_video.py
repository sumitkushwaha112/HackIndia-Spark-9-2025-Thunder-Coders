import cv2
import numpy as np

# Video writer setup
width, height = 64, 64
fps = 1
duration = 20  # seconds = frames (since fps = 1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sample_video.avi', fourcc, fps, (width, height))

# Create colored frames
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # RGB colors

for i in range(duration):
    color = colors[i % len(colors)]
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    out.write(frame)

out.release()
print("Sample video saved as sample_video.avi")
