import cv2
import numpy as np

def create_synthetic_concrete():
    # Base concrete texture
    np.random.seed(42)
    h, w = 600, 800
    base = np.random.normal(loc=180, scale=20, size=(h, w)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    
    # Concrete aggregate specks
    for _ in range(300):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(2, 6)
        color = int(np.random.randint(80, 140))
        cv2.circle(base, (cx, cy), r, color, -1)
        
    # Draw realistic jagged crack
    crack_mask = np.zeros((h, w), dtype=np.uint8)
    points = [(100, 150), (220, 200), (350, 260), (450, 290), (600, 420), (700, 480)]
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i+1]
        cv2.line(crack_mask, pt1, pt2, 255, thickness=4)
        
    # Add minor branch
    cv2.line(crack_mask, (350, 260), (420, 210), 255, thickness=3)

    # Perturb crack line for jagged look
    noise = np.random.randint(0, 2, size=(h, w), dtype=np.uint8) * 255
    crack_mask = cv2.bitwise_and(crack_mask, cv2.dilate(crack_mask, np.ones((3,3), np.uint8)))

    # Darken crack area on concrete
    base[crack_mask > 0] = np.clip(base[crack_mask > 0] * 0.35, 20, 60).astype(np.uint8)
    
    # Convert to RGB
    concrete_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("data/samples/sample_concrete_crack.jpg", concrete_rgb)
    cv2.imwrite("data/samples/sample_ground_truth_mask.png", crack_mask)
    print("Generated data/samples/sample_concrete_crack.jpg successfully.")

if __name__ == "__main__":
    create_synthetic_concrete()
