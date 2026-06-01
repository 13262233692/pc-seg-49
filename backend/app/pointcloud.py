import laspy
import numpy as np
import os
from werkzeug.utils import secure_filename

class PointCloudProcessor:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder

    def save_uploaded_file(self, file):
        filename = secure_filename(file.filename)
        filepath = os.path.join(self.upload_folder, filename)
        file.save(filepath)
        return filepath

    def load_las_file(self, filepath):
        las = laspy.read(filepath)
        
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack((las.red, las.green, las.blue)).transpose()
            if colors.max() > 1.0:
                colors = colors / 65535.0 if colors.max() > 255 else colors / 255.0
        else:
            colors = np.ones_like(points) * 0.7
        
        if hasattr(las, 'intensity'):
            intensity = las.intensity.reshape(-1, 1).astype(np.float64)
            p2 = np.percentile(intensity, 2)
            p98 = np.percentile(intensity, 98)
            if p98 - p2 > 1e-6:
                intensity = np.clip((intensity - p2) / (p98 - p2), 0, 1)
            else:
                intensity = np.ones_like(intensity) * 0.5
        else:
            intensity = np.ones((len(points), 1)) * 0.5
        
        return {
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'intensity': intensity.astype(np.float32),
            'num_points': len(points),
            'bounds': {
                'min': [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 2].min())],
                'max': [float(points[:, 0].max()), float(points[:, 1].max()), float(points[:, 2].max())]
            }
        }

    def sample_points(self, point_cloud_data, num_samples=50000):
        points = point_cloud_data['points']
        colors = point_cloud_data['colors']
        intensity = point_cloud_data['intensity']
        
        if len(points) <= num_samples:
            return point_cloud_data
        
        indices = np.random.choice(len(points), num_samples, replace=False)
        indices = np.sort(indices)
        
        return {
            'points': points[indices],
            'colors': colors[indices],
            'intensity': intensity[indices],
            'num_points': num_samples,
            'bounds': point_cloud_data['bounds'],
            'original_num_points': len(points)
        }

    def normalize_points(self, points):
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        return points_normalized, centroid, max_dist
