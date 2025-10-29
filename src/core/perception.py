#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)

    dest_points = [x["translation_vector"][:2] for x in hoop_positions]
    src_points = []
    
    for img in images:
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(gray_blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * (area/(perimeter*perimeter))
            if 0.7 < circularity <= 1.2:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 10:
                    center = (int(x), int(y))
                    src_points.append(center)
                    #cv2.circle(img, center, int(radius), (0, 255, 0), 2)
                    #cv2.circle(img, center, 2, (0, 0, 255), 3)            
                    #cv2.imshow("Detected Circles", img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    break     
    
    homography, mask = cv2.findHomography(np.array(src_points), np.array(dest_points))
    return homography