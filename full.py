import cv2 
import os 
import numpy as np 
import copy

def wall_bfs(map, x, y, color, h, w):
    queue = [(x, y)]
    min_xval = x
    max_xval = x
    min_yval = y
    max_yval = y

    while len(queue) > 0:
        x, y = queue.pop(0)
        if map[x][y] == color:
            continue
        map[x][y] = color

        min_xval = min(min_xval, x)
        max_xval = max(max_xval, x)
        min_yval = min(min_yval, y)
        max_yval = max(max_yval, y)

        if x > 0 and map[x - 1][y] == 1:
            queue.append((x - 1, y))
        if x < h - 1 and map[x + 1][y] == 1:
            queue.append((x + 1, y))
        if y > 0 and map[x][y - 1] == 1:
            queue.append((x, y - 1))
        if y < w - 1 and map[x][y + 1] == 1:
            queue.append((x, y + 1))
    
    
    if (max_xval - min_xval) * (max_yval - min_yval) < h * w * 0.005:
        for i in range(min_xval, max_xval + 1):
            for j in range(min_yval, max_yval + 1):
                if map[i][j] == color:
                    map[i][j] = 0

def bfs(map, x, y, color, h, w):
    queue = [(x, y)]
    area = 0 
    max_xval = x
    max_yval = y
    min_xval = x
    min_yval = y

    while len(queue) > 0:
        x, y = queue.pop(0)
        if map[x][y] == color:
            continue
        map[x][y] = color
        area += 1
        max_xval = max(max_xval, x)
        max_yval = max(max_yval, y)
        min_xval = min(min_xval, x)
        min_yval = min(min_yval, y)
        if x > 0 and map[x - 1][y] == 0:
            queue.append((x - 1, y))
        if x < h - 1 and map[x + 1][y] == 0:
            queue.append((x + 1, y))
        if y > 0 and map[x][y - 1] == 0:
            queue.append((x, y - 1))
        if y < w - 1 and map[x][y + 1] == 0:
            queue.append((x, y + 1))
    
    flag = True 
    if area < h * w * 0.002:
        flag = False 
        for i in range(h):
            for j in range(w):
                if map[i][j] == color:
                    map[i][j] = 3
    return flag, [min_xval, max_xval, min_yval, max_yval], area

def canny_edge_detection(image, low_threshold=25, high_threshold=25):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges

def full_pipeline(in_image, out_folder_path=None, debug=False):
    if debug:
        cv2.imwrite(os.path.join(out_folder_path, "original.jpg"), in_image)

    h, w, c = in_image.shape
    if debug:
        copyed_image = copy.deepcopy(in_image)

    # remove gray components
    for i in range(h):
        for j in range(w):
            r, g, b = in_image[i][j]
            if r == g and g == b and r > 160:
                in_image[i][j] = [255, 255, 255]
    
    if debug and not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    # save the image with gray components removed
    if debug:
        cv2.imwrite(os.path.join(out_folder_path, "gray_removed.jpg"), in_image)

    # edge detection
    edge_map = canny_edge_detection(in_image)
    if debug:
        cv2.imwrite(os.path.join(out_folder_path, "edge_map.jpg"), edge_map)

    edge_map = cv2.GaussianBlur(edge_map, (5, 5), 0)

    edge_map[edge_map > 0] = 255.
    if debug:
        cv2.imwrite(os.path.join(out_folder_path, "edge_map_blurred.jpg"), 255. - edge_map)

    map = [[1 if edge_map[j][i] > 100 else 0 for i in range(w)] for j in range(h)]

    outside = [0, 0]
    while map[outside[0]][outside[1]] == 1:
        outside[1] += 1
        outside[0] += 1

    #bfs(map, outside[0], outside[1], 3, h, w)

    outside = [h - 1, w - 1]

    while map[outside[0]][outside[1]] == 1:
        outside[0] -= 1
        outside[1] -= 1

    #bfs(map, outside[0], outside[1], 3, h, w)

    # -1 : outside
    # 0 : empty
    # 1 : wall
    # 2 : visited wall
    # 3 : small partitions, outside 
    
    color_map = {-1:(50, 50, 50), 0:(255, 255, 255), 1:(0, 0, 0), 2:(100, 100, 100), 3:(255, 255, 255)}

    for i in range(4, 100):
        color_map[i] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    
    def map_to_image(map):
        image = [[[0, 0, 0] for i in range(w)] for j in range(h)]
        for i in range(h):
            for j in range(w):
                image[i][j] = color_map[map[i][j]]
        return np.array(image)
    
    while True:
        found = False
        for i in range(h):
            for j in range(w):
                if map[i][j] == 1:
                    wall_bfs(map, i, j, 2, h, w)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    count = 3 
    result = []
    while True:
        found = False 
        for i in range(h):
            for j in range(w):
                if map[i][j] == 0:
                    res, bbox, area = bfs(map, i, j, count, h, w)
                    if res:
                        count += 1
                        result.append((bbox, area))
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    result = [( bbox, area) for bbox, area in result if (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) < h * w * 0.7 and bbox[0] != 0 and bbox[1] != 0 and bbox[2] != w - 1 and bbox[3] != h - 1]
    result.sort(key=lambda x: x[1])
    

    output = map_to_image(map)
    if debug:
        cv2.imwrite(os.path.join(out_folder_path, "output.jpg"), output)

    if debug:
        in_image = copyed_image

        for i in range(h):
            for j in range(w):
                if map[i][j] > 3 and sum(in_image[i][j]) > 200:
                    in_image[i][j] = in_image[i][j] * 0.5 + output[i][j] * 0.5

        for bbox, area in result:
            min_xval, max_xval, min_yval, max_yval = bbox
            cv2.rectangle(in_image, (min_yval, min_xval), (max_yval, max_xval), (0, 0, 255), 3)

        cv2.imwrite(os.path.join(out_folder_path, "output_with_original.jpg"), in_image)

    bboxes = [bbox for bbox, area in result]
    bboxes = [{"bbox":(bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2]), "level":idx} for idx, bbox in enumerate(bboxes)]

    return bboxes
    



if __name__ == '__main__':
    full_pipeline("test.jpeg", "output")