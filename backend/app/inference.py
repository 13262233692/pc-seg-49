import numpy as np
import os
import onnxruntime as ort
from scipy.spatial import cKDTree

CLASSES = {
    0: 'unclassified',
    1: 'ground',
    2: 'building',
    3: 'vegetation',
    4: 'vehicle',
    5: 'water'
}

CLASS_COLORS = {
    0: [128, 128, 128],
    1: [139, 69, 19],
    2: [112, 128, 144],
    3: [34, 139, 34],
    4: [255, 99, 71],
    5: [65, 105, 225]
}

NOISE_CLASS_IDS = {6, 7, 8, 9, 10, 65, 72, 73, 74, 75, 76, 77, 78, 79}


class KPConvInference:
    def __init__(self, models_folder):
        self.models_folder = models_folder
        self.model_path = os.path.join(models_folder, 'kpconv_model.onnx')
        self.session = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.session = ort.InferenceSession(self.model_path)
                print("KPConv model loaded successfully")
            except Exception as e:
                print(f"Failed to load KPConv model: {e}")
                self.session = None
        else:
            print("No pre-trained model found, using heuristic-based segmentation")
            self.session = None

    def _robust_intensity_normalize(self, intensity):
        if intensity is None or len(intensity) == 0:
            return None
        flat = intensity.flatten().astype(np.float64)
        p2 = np.percentile(flat, 2)
        p98 = np.percentile(flat, 98)
        if p98 - p2 < 1e-6:
            return np.ones_like(flat) * 0.5
        return np.clip((flat - p2) / (p98 - p2), 0, 1)

    def _compute_local_features(self, points, k=20):
        n = len(points)
        k = min(k + 1, n)
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k)

        mean_dist = np.mean(distances[:, 1:], axis=1)
        local_density = 1.0 / (mean_dist + 1e-6)

        neighbor_z = points[indices, 2]
        local_z_range = np.max(neighbor_z, axis=1) - np.min(neighbor_z, axis=1)
        local_z_std = np.std(neighbor_z, axis=1)

        neighbor_pts = points[indices]
        centroid_xy = np.mean(neighbor_pts[:, :, :2], axis=1)
        dx = neighbor_pts[:, :, 0] - centroid_xy[:, np.newaxis, 0]
        dy = neighbor_pts[:, :, 1] - centroid_xy[:, np.newaxis, 1]
        local_xy_spread = np.sqrt(np.mean(dx ** 2 + dy ** 2, axis=1))

        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min if z_max - z_min > 1e-6 else 1.0
        normalized_height = (z - z_min) / z_range

        return {
            'density': local_density,
            'z_range': local_z_range,
            'z_std': local_z_std,
            'xy_spread': local_xy_spread,
            'normalized_height': normalized_height,
            'tree': tree,
            'indices': indices
        }

    def _heuristic_segmentation(self, points, colors=None, intensity=None):
        num_points = len(points)
        labels = np.zeros(num_points, dtype=np.int32)
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min if z_max - z_min > 1e-6 else 1.0

        features = self._compute_local_features(points, k=20)
        norm_h = features['normalized_height']
        z_std = features['z_std']
        z_rng = features['z_range']
        xy_spread = features['xy_spread']

        robust_intensity = self._robust_intensity_normalize(intensity)

        ground_mask = norm_h < 0.12

        low_z_std = z_std < (z_range * 0.15)
        low_z_range = z_rng < (z_range * 0.3)
        not_ground = ~ground_mask

        building_mask = np.zeros(num_points, dtype=bool)

        flat_surface = not_ground & low_z_std & low_z_range
        building_mask |= flat_surface

        mid_high = (norm_h > 0.12) & (norm_h < 0.98)
        moderate_xy = xy_spread > np.percentile(xy_spread, 30)
        structured = not_ground & mid_high & moderate_xy & low_z_range
        building_mask |= structured

        if robust_intensity is not None:
            intensity_p75 = np.percentile(robust_intensity, 75)
            moderate_int = (robust_intensity > 0.25) & (robust_intensity <= intensity_p75)
            int_building = not_ground & mid_high & moderate_int
            building_mask |= int_building

            high_int = robust_intensity > intensity_p75
            building_like = not_ground & mid_high & (low_z_std | low_z_range)
            high_int_building = not_ground & high_int & building_like
            building_mask |= high_int_building

        building_mask &= ~ground_mask

        vegetation_mask = np.zeros(num_points, dtype=bool)
        if colors is not None:
            r = np.clip(colors[:, 0], 1e-6, None)
            g = colors[:, 1]
            b = np.clip(colors[:, 2], 1e-6, None)
            green_ratio = g / (r + b + 1e-6)
            ndvi_like = (g - r) / (g + r + 1e-6)
            vegetation_mask = (not_ground & ~building_mask &
                               ((green_ratio > 1.05) | (ndvi_like > 0.1)))

        labels[ground_mask] = 1
        labels[building_mask] = 2
        labels[vegetation_mask] = 3

        return labels, features

    def _postprocess_labels(self, points, labels, tree=None, k=20, min_ratio=0.35, iterations=2):
        n = len(points)
        if tree is None:
            tree = cKDTree(points)
        k = min(k + 1, n)
        _, indices = tree.query(points, k=k)

        new_labels = labels.copy()

        for iteration in range(iterations):
            changed = 0
            neighbor_labels = new_labels[indices[:, 1:]]

            for i in range(n):
                if new_labels[i] != 0:
                    continue
                n_labels = neighbor_labels[i]
                n_valid = n_labels[n_labels > 0]
                if len(n_valid) == 0:
                    continue
                unique, counts = np.unique(n_valid, return_counts=True)
                dominant_idx = np.argmax(counts)
                dominant_label = unique[dominant_idx]
                ratio = counts[dominant_idx] / (k - 1)
                if ratio >= min_ratio:
                    new_labels[i] = int(dominant_label)
                    changed += 1

            if changed == 0:
                break
            labels = new_labels.copy()

        return new_labels

    def _reclassify_high_reflectivity_building_holes(self, points, labels, intensity, tree=None, k=25):
        if intensity is None:
            return labels

        robust_intensity = self._robust_intensity_normalize(intensity)
        if robust_intensity is None:
            return labels

        new_labels = labels.copy()

        p90 = np.percentile(robust_intensity, 90)
        high_refl_mask = robust_intensity > p90

        potential_holes = high_refl_mask & ((labels == 0) | (labels == 1))
        if not np.any(potential_holes):
            return new_labels

        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min if z_max - z_min > 1e-6 else 1.0
        normalized_height = (z - z_min) / z_range
        above_ground = normalized_height > 0.12

        candidate_mask = potential_holes & above_ground
        if not np.any(candidate_mask):
            return new_labels

        if tree is None:
            tree = cKDTree(points)
        k = min(k + 1, len(points))
        candidate_indices = np.where(candidate_mask)[0]
        _, neighbor_indices = tree.query(points[candidate_indices], k=k)

        for local_idx, global_idx in enumerate(candidate_indices):
            n_idx = neighbor_indices[local_idx, 1:]
            n_labels = new_labels[n_idx]
            n_valid = n_labels[n_labels > 0]

            if len(n_valid) == 0:
                continue

            unique, counts = np.unique(n_valid, return_counts=True)
            building_count = counts[unique == 2].sum() if 2 in unique else 0
            total_valid = len(n_valid)

            building_ratio = building_count / total_valid if total_valid > 0 else 0

            if building_ratio > 0.3:
                new_labels[global_idx] = 2

        return new_labels

    def _handle_onnx_noise_labels(self, points, labels, k=15):
        noise_mask = np.isin(labels, list(NOISE_CLASS_IDS))
        unknown_mask = labels >= len(CLASSES)
        problem_mask = noise_mask | unknown_mask

        if not np.any(problem_mask):
            return labels

        new_labels = labels.copy()
        tree = cKDTree(points)
        k = min(k + 1, len(points))
        problem_indices = np.where(problem_mask)[0]

        if len(problem_indices) > 0:
            _, neighbor_indices = tree.query(points[problem_indices], k=k)

            for local_idx, global_idx in enumerate(problem_indices):
                n_idx = neighbor_indices[local_idx, 1:]
                n_labels = labels[n_idx]
                n_valid = n_labels[~np.isin(n_labels, list(NOISE_CLASS_IDS))]
                n_valid = n_valid[n_valid < len(CLASSES)]

                if len(n_valid) > 0:
                    unique, counts = np.unique(n_valid, return_counts=True)
                    dominant = unique[np.argmax(counts)]
                    new_labels[global_idx] = int(dominant)
                else:
                    new_labels[global_idx] = 2

        return new_labels

    def _knn_features(self, points, k=10):
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=min(k + 1, len(points)))

        local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
        local_z_std = np.std(points[indices][:, :, 2], axis=1)

        features = np.column_stack([
            local_density,
            local_z_std,
            points[:, 2] - np.min(points[:, 2])
        ])

        return features

    def segment(self, point_cloud_data):
        points = point_cloud_data['points']
        colors = point_cloud_data.get('colors')
        intensity = point_cloud_data.get('intensity')

        features = None

        if self.session is not None:
            try:
                normalized_points, _, _ = self._normalize_points(points)
                model_features = self._knn_features(normalized_points)

                input_tensor = np.concatenate([normalized_points, model_features], axis=1)
                input_tensor = input_tensor.astype(np.float32).reshape(1, -1, input_tensor.shape[1])

                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name

                result = self.session.run([output_name], {input_name: input_tensor})
                labels = np.argmax(result[0], axis=2).flatten()

                labels = self._handle_onnx_noise_labels(points, labels)

            except Exception as e:
                print(f"ONNX inference failed, falling back to heuristic: {e}")
                labels, features = self._heuristic_segmentation(points, colors, intensity)
        else:
            labels, features = self._heuristic_segmentation(points, colors, intensity)

        tree = features['tree'] if features else None

        labels = self._reclassify_high_reflectivity_building_holes(
            points, labels, intensity, tree=tree
        )

        labels = self._postprocess_labels(
            points, labels, tree=tree, k=20, min_ratio=0.35, iterations=2
        )

        label_colors = np.array([CLASS_COLORS.get(int(label), CLASS_COLORS[0]) for label in labels]) / 255.0

        class_distribution = {}
        for label in np.unique(labels):
            count = np.sum(labels == label)
            class_distribution[CLASSES.get(int(label), f'class_{label}')] = int(count)

        return {
            'labels': labels,
            'label_colors': label_colors,
            'class_distribution': class_distribution,
            'classes': CLASSES,
            'class_colors': {k: [c / 255.0 for c in v] for k, v in CLASS_COLORS.items()}
        }

    def _normalize_points(self, points):
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        return points_normalized, centroid, max_dist

    def get_class_colors(self):
        return {k: [c / 255.0 for c in v] for k, v in CLASS_COLORS.items()}

    def get_classes(self):
        return CLASSES
