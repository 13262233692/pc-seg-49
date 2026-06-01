from flask import request, jsonify
from app import app
from app.pointcloud import PointCloudProcessor
from app.inference import KPConvInference
from app.change_detection import ChangeDetector
import os
import numpy as np
import json
import uuid

processor = PointCloudProcessor(app.config['UPLOAD_FOLDER'])
inference = KPConvInference(app.config['MODELS_FOLDER'])
change_detector = ChangeDetector()

point_cloud_cache = {}
segmentation_cache = {}
point_cloud_cache_period2 = {}
segmentation_cache_period2 = {}
change_detection_cache = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference.session is not None
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': inference.get_classes(),
        'class_colors': inference.get_class_colors()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not (file.filename.endswith('.las') or file.filename.endswith('.laz')):
        return jsonify({'error': 'Only LAS/LAZ files are supported'}), 400
    
    try:
        filepath = processor.save_uploaded_file(file)
        point_cloud_data = processor.load_las_file(filepath)
        
        max_points = request.form.get('max_points', 50000, type=int)
        sampled_data = processor.sample_points(point_cloud_data, max_points)
        
        pc_id = str(uuid.uuid4())
        point_cloud_cache[pc_id] = sampled_data
        
        normalized_points, centroid, max_dist = processor.normalize_points(sampled_data['points'])
        
        response = {
            'id': pc_id,
            'filename': file.filename,
            'num_points': sampled_data['num_points'],
            'original_num_points': sampled_data.get('original_num_points', sampled_data['num_points']),
            'bounds': sampled_data['bounds'],
            'centroid': centroid.tolist(),
            'max_dist': float(max_dist),
            'points': normalized_points.tolist(),
            'colors': sampled_data['colors'].tolist(),
            'intensity': sampled_data['intensity'].flatten().tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment/<pc_id>', methods=['POST'])
def segment_point_cloud(pc_id):
    if pc_id not in point_cloud_cache:
        return jsonify({'error': 'Point cloud not found'}), 404
    
    try:
        point_cloud_data = point_cloud_cache[pc_id]
        segmentation_result = inference.segment(point_cloud_data)
        
        segmentation_cache[pc_id] = segmentation_result
        
        response = {
            'labels': segmentation_result['labels'].tolist(),
            'label_colors': segmentation_result['label_colors'].tolist(),
            'class_distribution': segmentation_result['class_distribution'],
            'classes': segmentation_result['classes'],
            'class_colors': segmentation_result['class_colors']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-labels/<pc_id>', methods=['POST'])
def update_labels(pc_id):
    if pc_id not in segmentation_cache:
        return jsonify({'error': 'Segmentation not found'}), 404
    
    try:
        data = request.json
        point_indices = data.get('point_indices', [])
        new_class = data.get('new_class', 0)
        
        if not point_indices:
            return jsonify({'error': 'No point indices provided'}), 400
        
        segmentation_result = segmentation_cache[pc_id]
        labels = segmentation_result['labels']
        class_colors = segmentation_result['class_colors']
        
        for idx in point_indices:
            if 0 <= idx < len(labels):
                labels[idx] = new_class
        
        new_colors = np.array([class_colors.get(str(label), class_colors.get(int(label), [0.5, 0.5, 0.5])) for label in labels])
        segmentation_result['label_colors'] = new_colors
        
        class_distribution = {}
        for label in np.unique(labels):
            count = np.sum(labels == label)
            class_name = inference.get_classes().get(int(label), f'class_{label}')
            class_distribution[class_name] = int(count)
        segmentation_result['class_distribution'] = class_distribution
        
        return jsonify({
            'labels': labels.tolist(),
            'label_colors': new_colors.tolist(),
            'class_distribution': class_distribution
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<pc_id>', methods=['GET'])
def export_results(pc_id):
    if pc_id not in point_cloud_cache or pc_id not in segmentation_cache:
        return jsonify({'error': 'Data not found'}), 404
    
    try:
        point_cloud_data = point_cloud_cache[pc_id]
        segmentation_result = segmentation_cache[pc_id]
        
        export_data = {
            'points': point_cloud_data['points'].tolist(),
            'colors': point_cloud_data['colors'].tolist(),
            'intensity': point_cloud_data['intensity'].tolist(),
            'labels': segmentation_result['labels'].tolist(),
            'label_colors': segmentation_result['label_colors'].tolist(),
            'class_distribution': segmentation_result['class_distribution']
        }
        
        return jsonify(export_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear/<pc_id>', methods=['DELETE'])
def clear_cache(pc_id):
    if pc_id in point_cloud_cache:
        del point_cloud_cache[pc_id]
    if pc_id in segmentation_cache:
        del segmentation_cache[pc_id]
    return jsonify({'status': 'success'})


@app.route('/api/change/types', methods=['GET'])
def get_change_types():
    return jsonify({
        'change_types': change_detector.get_change_types(),
        'change_colors': change_detector.get_change_colors()
    })


@app.route('/api/change/upload-period2', methods=['POST'])
def upload_period2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not (file.filename.endswith('.las') or file.filename.endswith('.laz')):
        return jsonify({'error': 'Only LAS/LAZ files are supported'}), 400
    
    try:
        filepath = processor.save_uploaded_file(file)
        point_cloud_data = processor.load_las_file(filepath)
        
        max_points = request.form.get('max_points', 50000, type=int)
        sampled_data = processor.sample_points(point_cloud_data, max_points)
        
        pc_id = str(uuid.uuid4())
        point_cloud_cache_period2[pc_id] = sampled_data
        
        normalized_points, centroid, max_dist = processor.normalize_points(sampled_data['points'])
        
        response = {
            'id': pc_id,
            'filename': file.filename,
            'num_points': sampled_data['num_points'],
            'original_num_points': sampled_data.get('original_num_points', sampled_data['num_points']),
            'bounds': sampled_data['bounds'],
            'centroid': centroid.tolist(),
            'max_dist': float(max_dist),
            'points': normalized_points.tolist(),
            'colors': sampled_data['colors'].tolist(),
            'intensity': sampled_data['intensity'].flatten().tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/change/segment-period2/<pc_id>', methods=['POST'])
def segment_period2(pc_id):
    if pc_id not in point_cloud_cache_period2:
        return jsonify({'error': 'Point cloud not found'}), 404
    
    try:
        point_cloud_data = point_cloud_cache_period2[pc_id]
        segmentation_result = inference.segment(point_cloud_data)
        
        segmentation_cache_period2[pc_id] = segmentation_result
        
        response = {
            'labels': segmentation_result['labels'].tolist(),
            'label_colors': segmentation_result['label_colors'].tolist(),
            'class_distribution': segmentation_result['class_distribution'],
            'classes': segmentation_result['classes'],
            'class_colors': segmentation_result['class_colors']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/change/detect/<pc1_id>/<pc2_id>', methods=['POST'])
def detect_changes(pc1_id, pc2_id):
    if pc1_id not in point_cloud_cache or pc1_id not in segmentation_cache:
        return jsonify({'error': 'Period 1 data not found'}), 404
    if pc2_id not in point_cloud_cache_period2 or pc2_id not in segmentation_cache_period2:
        return jsonify({'error': 'Period 2 data not found'}), 404
    
    try:
        data = request.json or {}
        distance_threshold = data.get('distance_threshold', 0.08)
        min_cluster_size = data.get('min_cluster_size', 20)
        
        pc1 = point_cloud_cache[pc1_id]
        pc2 = point_cloud_cache_period2[pc2_id]
        seg1 = segmentation_cache[pc1_id]
        seg2 = segmentation_cache_period2[pc2_id]
        
        change_result = change_detector.detect_changes(
            pc1, pc2, seg1, seg2,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size
        )
        
        change_id = str(uuid.uuid4())
        change_detection_cache[change_id] = {
            'pc1_id': pc1_id,
            'pc2_id': pc2_id,
            'result': change_result
        }
        
        change_result['change_id'] = change_id
        
        return jsonify(change_result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/change/export/<change_id>', methods=['GET'])
def export_changes(change_id):
    if change_id not in change_detection_cache:
        return jsonify({'error': 'Change detection result not found'}), 404
    
    try:
        data = change_detection_cache[change_id]
        
        export_data = {
            'period1': {
                'id': data['pc1_id'],
                'points': point_cloud_cache[data['pc1_id']]['points'].tolist(),
                'labels': segmentation_cache[data['pc1_id']]['labels'].tolist(),
                'change_labels': data['result']['period1']['change_labels'],
            },
            'period2': {
                'id': data['pc2_id'],
                'points': point_cloud_cache_period2[data['pc2_id']]['points'].tolist(),
                'labels': segmentation_cache_period2[data['pc2_id']]['labels'].tolist(),
                'change_labels': data['result']['period2']['change_labels'],
            },
            'statistics': data['result']['statistics'],
            'change_types': data['result']['change_types']
        }
        
        return jsonify(export_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/change/clear-all', methods=['DELETE'])
def clear_change_data():
    point_cloud_cache_period2.clear()
    segmentation_cache_period2.clear()
    change_detection_cache.clear()
    return jsonify({'status': 'success'})
