from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from image_processing import preprocess_image, preprocess_colour_image, intensity_to_params

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    if file.filename == '':
        return 'No file selected', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return render_template('index.html',
                             uploaded=True,
                             filename=filename,
                             original_path=f'/uploads/{filename}')

    return 'Invalid file type', 400


@app.route('/process', methods=['POST'])
def process_image():
    """Process the uploaded image"""
    filename = request.form.get('filename')

    if not filename:
        return 'No filename provided', 400

    # Get processing options (checkboxes)
    use_clahe = request.form.get('use_clahe') == 'on'
    use_illum = request.form.get('use_illum') == 'on'
    use_rolling = request.form.get('use_rolling') == 'on'
    use_gauss = request.form.get('use_gauss') == 'on'
    use_median = request.form.get('use_median') == 'on'
    use_bilateral = request.form.get('use_bilateral') == 'on'

    # Get intensity values (0-100 from sliders)
    clahe_intensity = int(request.form.get('clahe_intensity', 50))
    illum_intensity = int(request.form.get('illum_intensity', 50))
    rolling_intensity = int(request.form.get('rolling_intensity', 50))
    gauss_intensity = int(request.form.get('gauss_intensity', 50))
    median_intensity = int(request.form.get('median_intensity', 50))
    bilateral_intensity = int(request.form.get('bilateral_intensity', 50))

    # Load image - ALWAYS from original upload
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return 'Could not load image', 400

    # Check if any processing is requested
    any_processing = use_clahe or use_illum or use_rolling or use_gauss or use_median or use_bilateral

    if not any_processing:
        # No processing - return to original state (no processed image)
        form_state = {
            'use_clahe': False,
            'use_illum': False,
            'use_rolling': False,
            'use_gauss': False,
            'use_median': False,
            'use_bilateral': False,
            'clahe_intensity': clahe_intensity,
            'illum_intensity': illum_intensity,
            'rolling_intensity': rolling_intensity,
            'gauss_intensity': gauss_intensity,
            'median_intensity': median_intensity,
            'bilateral_intensity': bilateral_intensity
        }

        return render_template('index.html',
                             uploaded=True,
                             processed=False,
                             filename=filename,
                             original_path=f'/uploads/{filename}',
                             form_state=form_state)

    # Convert intensity values (0-100) to actual parameter ranges
    custom_params = intensity_to_params(
        img,
        clahe_intensity,
        illum_intensity,
        rolling_intensity,
        gauss_intensity,
        median_intensity,
        bilateral_intensity
    )

    # Process image (handles both color and grayscale)
    # IMPORTANT: Always processes from original image, not from previous processed result
    if len(img.shape) == 3:
        # Color image - use color processing pipeline
        processed, metrics_before, metrics_after, params = preprocess_colour_image(
            img,
            use_clahe=use_clahe,
            use_illum=use_illum,
            use_rolling=use_rolling,
            use_gauss=use_gauss,
            use_median=use_median,
            use_bilateral=use_bilateral,
            auto_params=False,
            params=custom_params
        )
    else:
        # Grayscale image - use standard pipeline
        processed, metrics_before, metrics_after, params = preprocess_image(
            img,
            use_clahe=use_clahe,
            use_illum=use_illum,
            use_rolling=use_rolling,
            use_gauss=use_gauss,
            use_median=use_median,
            use_bilateral=use_bilateral,
            auto_params=False,
            params=custom_params
        )

    # Save processed image
    processed_filename = f'processed_{filename}'
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(output_path, processed)

    # Preserve form state for persistent settings
    form_state = {
        'use_clahe': use_clahe,
        'use_illum': use_illum,
        'use_rolling': use_rolling,
        'use_gauss': use_gauss,
        'use_median': use_median,
        'use_bilateral': use_bilateral,
        'clahe_intensity': clahe_intensity,
        'illum_intensity': illum_intensity,
        'rolling_intensity': rolling_intensity,
        'gauss_intensity': gauss_intensity,
        'median_intensity': median_intensity,
        'bilateral_intensity': bilateral_intensity
    }

    return render_template('index.html',
                         uploaded=True,
                         processed=True,
                         filename=filename,
                         processed_filename=processed_filename,
                         original_path=f'/uploads/{filename}',
                         processed_path=f'/processed/{processed_filename}',
                         metrics_before=metrics_before,
                         metrics_after=metrics_after,
                         params=params,
                         form_state=form_state)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/processed/<filename>')
def processed_file(filename):
    """Serve processed files"""
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))


@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    return send_file(
        os.path.join(app.config['PROCESSED_FOLDER'], filename),
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True, port=7400)
