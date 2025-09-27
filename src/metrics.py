from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Model prediction duration')
ACTIVE_PREDICTIONS = Gauge('active_predictions', 'Number of active predictions')
MODEL_PREDICTIONS_TOTAL = Counter('model_predictions_total', 'Total predictions made')
ERROR_COUNT = Counter('api_errors_total', 'Total API errors', ['error_type'])

def track_requests(f):
    """Decorator to track request metrics"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
        ACTIVE_PREDICTIONS.inc()
        
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            ACTIVE_PREDICTIONS.dec()
    
    return wrapper

def track_prediction_time(f):
    """Decorator to track model prediction time"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            MODEL_PREDICTIONS_TOTAL.inc()
            return result
        finally:
            duration = time.time() - start_time
            PREDICTION_DURATION.observe(duration)
    
    return wrapper
