from flask import Flask, request, jsonify
from celery import Celery
from flask_cors import CORS
import redis
import numpy as np

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configure Celery with Redis broker
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # Update if Redis is remote
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Grade mapping
grade_map = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
             'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'F': 0.0, 'WU': 0.0, 'W': 0.0}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        grades = [grade_map.get(data.get(f'c{i+1}'), 0.0) for i in range(11)]
        task = run_prediction.delay(grades)
        return jsonify({'task_id': task.id}), 202
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = run_prediction.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'status': 'Processing...'})
    elif task.state == 'SUCCESS':
        return jsonify({'cgpa': round(float(task.result), 2)})
    else:
        return jsonify({'status': task.state, 'error': str(task.info)})

# Celery Task
@celery.task(bind=True)
def run_prediction(self, grades):
    import pickle
    model = pickle.load(open('model/first_gpa.pkl', 'rb'))
    prediction = model.predict(np.array(grades).reshape(1, -1))
    return prediction[0][0]


if __name__ == '__main__':
    app.run(debug=True)
