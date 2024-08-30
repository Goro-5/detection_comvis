import cv2
import numpy as np
from deap import base, creator, tools, algorithms

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def get_largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_max = 0
    largest_cnt = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > area_max:
            area_max = w * h
            largest_cnt = cnt
    return largest_cnt

def draw_contour_and_boundary(image, contour):
    cv2.drawContours(image, [contour], 0, (255, 0, 0), 3)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    if np.linalg.norm(box[0]-box[1])>np.linalg.norm(box[1]-box[2]):
        box = box[1:] + box[:1]
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    return box

def angle_ratio_and_vector(pt0,pt1,pt2,pt3):
    vector1 = np.array(pt1) - np.array(pt0)
    vector2 = np.array(pt3) - np.array(pt2)
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    cos_theta = np.dot(vector1, vector2) / (length1 * length2)
    angle = np.arccos(cos_theta)
    ratio = length2 / length1
    vector = np.int32(((np.array(pt3) + np.array(pt2)) - (np.array(pt1) + np.array(pt0)))/2)
    return angle, ratio, vector

def transform_points(points, s, theta, tx, ty, center=(960,840)):
    # 中心座標を基準にポイントを移動
    centered_points = points - np.array(center)

    # 回転・拡大縮小行列の作成
    rotation_matrix = np.array([
        [s * np.cos(theta), -s * np.sin(theta)],
        [s * np.sin(theta), s * np.cos(theta)]
    ])

    # 回転・拡大縮小を適用
    transformed_pts = np.dot(centered_points, rotation_matrix.T)

    # ポイントを元の位置に戻す
    transformed_pts += np.array(center)

    # 平行移動を適用
    transformed_pts += np.array([tx, ty])

    return transformed_pts

def total_distance(s, theta, tx, ty, cnt1, cnt2, center=(960,840), origin=None):
    transformed_cnt1 = transform_points(cnt1.reshape(-1,2), s, theta, tx, ty, center)
    if origin is not None:
        origin2 = origin.copy()
        cv2.drawContours(origin2, [np.int32(transformed_cnt1)], 0, (0,255,255), 3)
        draw_contour_and_boundary(origin2, cnt2)
        cv2.imshow('origin', origin2)
        cv2.waitKey(1)
    distances = [np.min(np.linalg.norm(transformed_cnt1 - point, axis=1)) for point in cnt2.reshape(-1, 2)]
    return float(np.sum(distances))

def calculate_total_distance(individual, cnt1, cnt2, center, origin=None):
    s, theta, tx, ty = individual
    sum_distance = total_distance(s, theta, tx, ty, cnt1, cnt2, center, origin)
    return (sum_distance,)

def optimize_transform(cnt1, cnt2, center, origin=None):

    toolbox = base.Toolbox()
    toolbox.register("attr_tx", np.random.normal, 0, 640)
    toolbox.register("attr_ty", np.random.normal, 0, 560)
    toolbox.register("attr_theta", np.random.normal, 0, np.pi)  # θは-πからπの範囲
    toolbox.register("attr_s", np.random.normal, 0.7, 1.5)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_s, toolbox.attr_theta, toolbox.attr_tx, toolbox.attr_ty), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.6)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", calculate_total_distance, cnt1=cnt1, cnt2=cnt2, center=center, origin=origin)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=400, stats=stats, halloffame=hof, verbose=True)
    best_individual = hof[0]
    return best_individual[0], best_individual[1], best_individual[2], best_individual[3]

# 画像を読み込む
image_path = r"T:\Goro\ComputerVision\CamBookRaw\IMG_0009.png"
origin_path = r"T:\Goro\ComputerVision\joints_data\generated\PS-18SU\BACK\000.png"
image = cv2.imread(image_path)
frame = cv2.resize(image, (1920, 1680))
origin = cv2.imread(origin_path)

# 最大の輪郭を取得
largest_cnt = get_largest_contour(frame)
largest_origin_cnt = get_largest_contour(origin)

# 輪郭とQRコードを描画
boundary = draw_contour_and_boundary(frame, largest_cnt)
origin_boundary = draw_contour_and_boundary(origin, largest_origin_cnt)
# エラーハンドリングを追加
if len(origin_boundary) == 3:
    # 3点の場合、4点目を追加
    origin_boundary = np.vstack([origin_boundary, origin_boundary[0]])
elif len(origin_boundary) != 4:
    print(f"Warning: Unexpected number of boundary points: {len(origin_boundary)}")
    # 適切なデフォルト値を設定するか、処理を中断する


# 緑の枠線の新しい位置を計算
qr = cv2.QRCodeDetector()
data, points, straight_qrcode = qr.detectAndDecode(frame)
if points is not None:
    points = points.astype(np.int32)
    cv2.polylines(frame, [points], True, (0, 255, 0), thickness=10)

cent_pt_sub = np.int32((np.array(origin_boundary[0]) + np.array(origin_boundary[1])) / 2)
prm1_theta, prm1_s, prm1_t = angle_ratio_and_vector(boundary[1], boundary[2], origin_boundary[1], origin_boundary[2])
prm2_theta, prm2_s, prm2_t = angle_ratio_and_vector(boundary[3], boundary[0], origin_boundary[1], origin_boundary[2])
dist1 = total_distance(prm1_s, prm1_theta, prm1_t[0], prm1_t[1], largest_cnt, largest_origin_cnt, cent_pt_sub, origin)
dist2 = total_distance(prm2_s, prm2_theta, prm2_t[0], prm2_t[1], largest_cnt, largest_origin_cnt, cent_pt_sub, origin)
if dist1 < dist2:
    prm_theta, prm_s, prm_t = prm1_theta, prm1_s, prm1_t
    print(dist1)
else:
    prm_theta, prm_s, prm_t = prm2_theta, prm2_s, prm2_t
    print(dist2)

cent_pt = (np.array(origin_boundary[0]) + np.array(origin_boundary[1]) + np.array(origin_boundary[2]) + np.array(origin_boundary[3])) / 4

largest_cnt_edited = transform_points(largest_cnt.reshape(-1, 2),prm_s,prm_theta,prm_t[0],prm_t[1])
cv2.drawContours(origin,[np.int32(largest_cnt_edited)],0,(0,0,0),3)
cv2.imshow('origin',origin)
# 座標変換パラメータを取得
s, theta, tx, ty = optimize_transform(largest_cnt_edited, largest_origin_cnt, cent_pt, origin)

# 緑の枠線の新しい位置を計算
if points is not None:
    transformed_points = transform_points(points.reshape(-1, 2), s, theta, tx, ty)
    cv2.polylines(origin, [np.int32(transformed_points)], True, (0, 255, 255), thickness=3)

# cnt1を変形させてoriginに重ねる
transformed_cnt1 = transform_points(largest_cnt_edited.reshape(-1, 2), s, theta, tx, ty)
cv2.drawContours(origin, [np.int32(transformed_cnt1)], 0, (0, 255, 0), 3)

# 結果を表示
cv2.imshow('frame', frame)
cv2.imshow('origin', origin)
cv2.waitKey(0)
cv2.destroyAllWindows()