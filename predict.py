import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# scored_y = pd.DataFrame([
#     # {"t": 1, "y": 1000},
#     {"t": 2, "y": 725},
#     {"t": 3, "y": 468},
#     {"t": 4, "y": 337}
# ])

# scored_x = pd.DataFrame([
#     # {"t": 1, "x": 1430},
#     {"t": 2, "x": 1368},
#     {"t": 3, "x": 1100},
#     {"t": 4, "x": 874}
# ])

# missed_y = pd.DataFrame([
#     {"t": 1, "y": 711},
#     {"t": 2, "y": 580},
#     {"t": 3, "y": 541},
#     {"t": 4, "y": 562}
# ])

# missed_x = pd.DataFrame([
#     {"t": 1, "x": 1331},
#     {"t": 2, "x": 1118},
#     {"t": 3, "x": 920},
#     {"t": 4, "x": 726},
# ])

frames_scored = [
    # r"C:\Users\austi\Downloads\ball motion\ball motion\scored\i_8452_00050.png",
    r"C:\Users\austi\Downloads\ball motion\ball motion\scored\i_8452_00051.png",
    r"C:\Users\austi\Downloads\ball motion\ball motion\scored\i_8452_00052.png",
    r"C:\Users\austi\Downloads\ball motion\ball motion\scored\i_8452_00053.png"
]

frames_missed = [
    r"C:\Users\austi\Downloads\ball motion\ball motion\missed\i_8452_00131.png",
    r"C:\Users\austi\Downloads\ball motion\ball motion\missed\i_8452_00132.png",
    r"C:\Users\austi\Downloads\ball motion\ball motion\missed\i_8452_00133.png",
    r"C:\Users\austi\Downloads\ball motion\ball motion\missed\i_8452_00134.png",
]

blue_lower = np.array([90, 50, 100], np.uint8)
blue_upper = np.array([150, 255, 255], np.uint8)

points_x = []
points_y = []

for i, file_path in enumerate(frames_missed):
    img = cv2.imread(file_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    contours, _ = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sorted(contours, key = lambda x: cv2.contourArea(x), reverse=True):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, _, _ = cv2.boundingRect(contour)
            points_x.append(x)
            points_y.append(y)
            break

points_y = pd.DataFrame(zip(points_y, list(np.arange(1,4))), columns=["y", "t"])
points_x = pd.DataFrame(zip(points_x, list(np.arange(1,4))), columns=["x", "t"])

m_y = smf.ols(formula="y ~ t + np.power(t, 2)", data=points_y).fit()
m_x = smf.ols(formula="x ~ t", data=points_x).fit()
print(m_y.summary())
print(m_x.summary())

pred_x = []
pred_y = []
for t in range(1, 11):
    y = m_y.params["np.power(t, 2)"]*t**2 + m_y.params["t"]*t + m_y.params["Intercept"]
    x = m_x.params["t"]*t + m_x.params["Intercept"]
    pred_x.append(x)
    pred_y.append(y)

plt.scatter(x=points_x["x"], y=points_y["y"])
plt.plot(pred_x, pred_y)
plt.plot([180, 160, 535, 500, 180], [450, 570, 520, 590, 450])
plt.gca().invert_yaxis()
plt.show()
print()
