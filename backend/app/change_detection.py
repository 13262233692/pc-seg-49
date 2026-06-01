import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import convolve1d


CHANGE_TYPES = {
    0: 'unchanged',
    1: 'new_building',
    2: 'demolished',
    3: 'ground',
    4: 'vegetation'
}

CHANGE_COLORS = {
    0: [0.6, 0.6, 0.6],
    1: [0.2, 1.0, 0.2],
    2: [1.0, 0.2, 0.2],
    3: [0.545, 0.271, 0.075],
    4: [0.133, 0.545, 0.133]
}


class ChangeDetector:
    def __init__(self):
        pass

    def _normalize_point_cloud(self, points):
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        return points_normalized, centroid, max_dist

    def _best_fit_transform(self, A, B):
        assert A.shape == B.shape

        m = A.shape[0]

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_B.T - np.dot(R, centroid_A.T)

        return R, t

    def _icp_registration(self, source, target, max_iterations=30, tolerance=1e-7):
        src = source.copy()
        prev_error = 0
        total_R = np.eye(3)
        total_t = np.zeros(3)

        tree = cKDTree(target)

        for i in range(max_iterations):
            distances, indices = tree.query(src, k=1)
            matched_target = target[indices]

            R, t = self._best_fit_transform(src, matched_target)

            src = np.dot(src, R.T) + t

            total_R = np.dot(R, total_R)
            total_t = np.dot(R, total_t) + t

            mean_error = np.mean(distances)
            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        return total_R, total_t

    def _get_building_mask(self, points, labels):
        return labels == 2

    def _compute_point_density(self, points, k=10):
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k + 1)
        mean_dist = np.mean(distances[:, 1:], axis=1)
        density = 1.0 / (mean_dist + 1e-6)
        return density

    def detect_changes(self, pc1_data, pc2_data, seg1, seg2, 
                       distance_threshold=0.08, min_cluster_size=20):
        points1 = pc1_data['points'].astype(np.float64)
        points2 = pc2_data['points'].astype(np.float64)
        labels1 = seg1['labels']
        labels2 = seg2['labels']

        norm_points1, centroid1, max_dist1 = self._normalize_point_cloud(points1)
        norm_points2, centroid2, max_dist2 = self._normalize_point_cloud(points2)

        n1 = len(points1)
        n2 = len(points2)

        try:
            sample_size = min(n1, n2, 20000)
            idx1 = np.random.choice(n1, sample_size, replace=False)
            idx2 = np.random.choice(n2, sample_size, replace=False)
            R, t = self._icp_registration(
                norm_points1[idx1], 
                norm_points2[idx2]
            )
            norm_points1_aligned = np.dot(norm_points1, R.T) + t
            points1_aligned = norm_points1_aligned * max_dist2 + centroid2
        except Exception as e:
            print(f"ICP registration failed: {e}, using raw coordinates")
            points1_aligned = points1

        building_mask1 = self._get_building_mask(points1, labels1)
        building_mask2 = self._get_building_mask(points2, labels2)

        change_labels1 = np.zeros(n1, dtype=np.int32)
        change_labels2 = np.zeros(n2, dtype=np.int32)

        change_labels1[labels1 == 1] = 3
        change_labels1[labels1 == 3] = 4
        change_labels1[building_mask1] = 0

        change_labels2[labels2 == 1] = 3
        change_labels2[labels2 == 3] = 4
        change_labels2[building_mask2] = 0

        tree2 = cKDTree(points2)

        if np.any(building_mask1):
            building_points1 = points1_aligned[building_mask1]
            distances, _ = tree2.query(building_points1, k=1)

            orig_indices1 = np.where(building_mask1)[0]
            demolished_mask = distances > distance_threshold

            for local_idx, global_idx in enumerate(orig_indices1):
                if demolished_mask[local_idx]:
                    change_labels1[global_idx] = 2

        tree1 = cKDTree(points1_aligned)

        if np.any(building_mask2):
            building_points2 = points2[building_mask2]
            distances, _ = tree1.query(building_points2, k=1)

            orig_indices2 = np.where(building_mask2)[0]
            new_mask = distances > distance_threshold

            for local_idx, global_idx in enumerate(orig_indices2):
                if new_mask[local_idx]:
                    change_labels2[global_idx] = 1

        if np.any(building_mask1) and np.any(building_mask2):
            orig_indices1 = np.where(building_mask1)[0]
            orig_indices2 = np.where(building_mask2)[0]

            for idx in orig_indices1:
                if change_labels1[idx] != 2:
                    dist, _ = tree2.query(points1_aligned[idx], k=1)
                    if dist <= distance_threshold:
                        change_labels1[idx] = 0

            for idx in orig_indices2:
                if change_labels2[idx] != 1:
                    dist, _ = tree1.query(points2[idx], k=1)
                    if dist <= distance_threshold:
                        change_labels2[idx] = 0

        new_count = int(np.sum(change_labels2 == 1))
        demolished_count = int(np.sum(change_labels1 == 2))

        new_points = points2[change_labels2 == 1]
        demolished_points = points1[change_labels1 == 2]

        new_area = 0.0
        if len(new_points) >= 3:
            xy_area = self._estimate_footprint_area(new_points)
            new_area = float(xy_area)

        demolished_area = 0.0
        if len(demolished_points) >= 3:
            xy_area = self._estimate_footprint_area(demolished_points)
            demolished_area = float(xy_area)

        unchanged_count1 = int(np.sum(change_labels1 == 0))
        unchanged_count2 = int(np.sum(change_labels2 == 0))

        change_colors1 = np.array([CHANGE_COLORS[label] for label in change_labels1])
        change_colors2 = np.array([CHANGE_COLORS[label] for label in change_labels2])

        result = {
            'period1': {
                'change_labels': change_labels1.tolist(),
                'change_colors': change_colors1.tolist(),
                'points': norm_points1_aligned.tolist() if 'norm_points1_aligned' in locals() else pc1_data['points'],
            },
            'period2': {
                'change_labels': change_labels2.tolist(),
                'change_colors': change_colors2.tolist(),
                'points': pc2_data['points'],
            },
            'statistics': {
                'new_building_count': new_count,
                'demolished_count': demolished_count,
                'new_building_area': new_area,
                'demolished_area': demolished_area,
                'unchanged_count_period1': unchanged_count1,
                'unchanged_count_period2': unchanged_count2,
                'distance_threshold': distance_threshold,
            },
            'change_types': CHANGE_TYPES,
            'change_colors': CHANGE_COLORS,
        }

        return result

    def _estimate_footprint_area(self, points):
        if len(points) < 3:
            return 0.0

        xy = points[:, :2]

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy)
            return float(hull.area)
        except:
            min_x, max_x = np.min(xy[:, 0]), np.max(xy[:, 0])
            min_y, max_y = np.min(xy[:, 1]), np.max(xy[:, 1])
            return float((max_x - min_x) * (max_y - min_y))

    def get_change_types(self):
        return CHANGE_TYPES

    def get_change_colors(self):
        return CHANGE_COLORS
